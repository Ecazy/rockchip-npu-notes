// SPDX-License-Identifier: GPL-3.0-or-later
/* Tests the real execute_impl loop; only BO operations and the blocking submit
 * ioctl are replaced.  In particular this target does not wrap co_execute*. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "custom_ops.h"
#include "mock_npu.h"

enum { TEST_FD = 3 };
static unsigned checks, failures, submits;

#define EXPECT(e) do { checks++; if (!(e)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #e); failures++; } } while (0)

/* Stub only the device submission boundary.  The production command builder,
 * submit call site, CO_SEPARATE recursion, and output synchronization all run. */
int __wrap_rocket_submit_tasks_flags(int fd, const rocket_task_desc *tasks,
                                     uint32_t n_tasks, const uint32_t *inputs,
                                     uint32_t ninputs, const uint32_t *outputs,
                                     uint32_t noutputs, uint32_t flags) {
    (void)fd; (void)tasks; (void)n_tasks; (void)inputs; (void)ninputs;
    (void)outputs; (void)noutputs; (void)flags;
    submits++;
    return 0;
}

typedef struct { co_buffer buffer; co_view first, second; int live; } shared_output;

static int make_matrix(co_buffer *buffer, co_view *view, unsigned rows,
                       unsigned cols, size_t offset) {
    co_tensor tensor;
    int rc=co_tensor_matrix(&tensor,rows,cols,CO_F16);
    if(rc)return rc;
    size_t payload=offset+tensor.bytes;
    if(offset==0)payload=tensor.bytes;
    rc=co_buffer_alloc(TEST_FD,payload,buffer);
    if(rc)return rc;
    rc=co_view_init(view,buffer,offset,&tensor);
    if(rc)co_buffer_free(buffer);
    return rc;
}

static int make_shared_output(shared_output *out, unsigned rows, unsigned cols) {
    co_tensor tensor;
    int rc=co_tensor_matrix(&tensor,rows,cols,CO_F16);
    if(rc)return rc;
    const size_t first_offset=32u;
    const size_t second_offset=first_offset+tensor.bytes;
    rc=co_buffer_alloc(TEST_FD,second_offset+tensor.bytes,&out->buffer);
    if(rc)return rc;
    out->live=1;
    rc=co_view_init(&out->first,&out->buffer,first_offset,&tensor);
    if(!rc)rc=co_view_init(&out->second,&out->buffer,second_offset,&tensor);
    if(rc){co_buffer_free(&out->buffer);out->live=0;}
    return rc;
}

static void release_shared(shared_output *out) {
    if(out->live)co_buffer_free(&out->buffer);
    memset(out,0,sizeof(*out));
}

/* fail_phase: 0 success, 1 fail first output PREP, 2 fail first output FINI. */
static void execute_case(enum co_mode mode, int same_bo, int fail_phase) {
    enum { ROWS=16, COLS=32 };
    co_buffer input_buffer, output_a_buffer, output_b_buffer;
    co_view input, output_a, output_b;
    shared_output shared;
    memset(&input_buffer,0,sizeof(input_buffer));
    memset(&output_a_buffer,0,sizeof(output_a_buffer));
    memset(&output_b_buffer,0,sizeof(output_b_buffer));
    memset(&input,0,sizeof(input));memset(&output_a,0,sizeof(output_a));
    memset(&output_b,0,sizeof(output_b));memset(&shared,0,sizeof(shared));

    int rc=make_matrix(&input_buffer,&input,ROWS,COLS,32u);
    if(!rc&&same_bo)rc=make_shared_output(&shared,ROWS,COLS);
    if(!rc&&!same_bo)rc=make_matrix(&output_a_buffer,&output_a,ROWS,COLS,32u);
    if(!rc&&!same_bo)rc=make_matrix(&output_b_buffer,&output_b,ROWS,COLS,32u);
    EXPECT(rc==0);
    if(rc)goto cleanup;
    if(same_bo){output_a=shared.first;output_b=shared.second;}

    co_program programs[2];
    memset(programs,0,sizeof(programs));
    EXPECT(co_emit_lut(&programs[0],input,output_a,CO_SIGMOID)==0);
    EXPECT(co_emit_lut(&programs[1],input,output_b,CO_SIGMOID)==0);

    rk_mock_reset();
    submits=0;
    if(fail_phase==1)rk_mock_fail_prep_at(2); /* command prep is call 1 */
    if(fail_phase==2)rk_mock_fail_fini_at(2); /* command fini is call 1 */
    rc=co_execute(TEST_FD,programs,2,mode);

    const unsigned nsubmits=mode==CO_SEPARATE?2u:1u;
#ifdef CO_OUTPUT_SYNC_REFERENCE
    const unsigned output_pairs=2u;
#else
    const unsigned output_pairs=(mode==CO_CHAIN&&same_bo)?1u:2u;
#endif
    EXPECT(submits==nsubmits);
    if(!fail_phase) {
        EXPECT(rc==0);
        EXPECT(rk_mock_prep_calls()==nsubmits+output_pairs);
        EXPECT(rk_mock_fini_calls()==nsubmits+output_pairs);
    } else if(fail_phase==1) {
        EXPECT(rc==ROCKET_E_DEVICE);
        EXPECT(rk_mock_prep_calls()==nsubmits+1u);
        EXPECT(rk_mock_fini_calls()==nsubmits);
    } else {
        EXPECT(rc==ROCKET_E_DEVICE);
        EXPECT(rk_mock_prep_calls()==nsubmits+1u);
        EXPECT(rk_mock_fini_calls()==nsubmits+1u);
    }
    rk_mock_reset();

cleanup:
    if(shared.live)release_shared(&shared);
    else {
        if(output_a_buffer.bo.handle)co_buffer_free(&output_a_buffer);
        if(output_b_buffer.bo.handle)co_buffer_free(&output_b_buffer);
    }
    if(input_buffer.bo.handle)co_buffer_free(&input_buffer);
    EXPECT(rk_mock_live_bos()==0);
}

int main(void) {
    execute_case(CO_CHAIN,1,0);       /* two output views, one BO, one submit */
    execute_case(CO_CHAIN,0,0);       /* two distinct output BOs */
    execute_case(CO_SEPARATE,1,0);    /* same BO, sync separately per submit */
    execute_case(CO_CHAIN,0,1);      /* first output PREP fails; stop */
    execute_case(CO_CHAIN,0,2);      /* first output FINI fails; stop */
    fprintf(stderr,"OUTPUT_SYNC checks=%u failures=%u\n",checks,failures);
    return failures?1:0;
}
