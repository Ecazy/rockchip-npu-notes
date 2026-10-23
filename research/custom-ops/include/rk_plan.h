// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef RK_PLAN_H
#define RK_PLAN_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RK_ABI_VERSION 1u
#define RK_SCHEMA_VERSION 2u
#define RK_MAX_SUBMIT_TASKS 64u
#define RK_MAX_PROGRAMS_PER_SUBMIT 64u
#define RK_MAX_WORDS_PER_PROGRAM 2048u
/* MVP verifier complexity bounds.  Real plans stay far below these; the limits
 * keep the pairwise alias/lifetime checks bounded for hostile descriptors. */
#define RK_MAX_WRITERS_PER_BUFFER 64u
/* Upper bound on inputs/outputs/deps an action may declare. */
#define RK_MAX_ACTION_ARGS 8u
/* Declared dependencies can cover every host action of a tile. */
#define RK_MAX_ACTION_DEPS 64u
#define RK_MAX_BUFFERS_PER_SLOT 64u
#define RK_DIAG_STAGE_LEN 32u
#define RK_DIAG_MESSAGE_LEN 512u
#define RK_DIAG_SOURCE_LEN 128u
#define RK_DIAG_PROFILE_LEN 64u
#define RK_ORIGIN_LEN 64u

enum rk_status {
    RK_E_INVALID = -1,
    RK_E_UNSUPPORTED = -2,
    RK_E_SHAPE = -3,
    RK_E_NUMERIC_CONTRACT = -4,
    RK_E_BUDGET = -5,
    RK_E_TARGET_MISMATCH = -6,
    RK_E_ALIAS = -7,
    RK_E_NOMEM = -8,
    RK_E_DEVICE = -9,
    RK_E_BUSY = -10,
    RK_E_POISONED = -11,
    RK_E_SCHEMA = -12,
    RK_E_DOMAIN = -13
};

enum rk_role {
    RK_ROLE_INPUT = 0,
    RK_ROLE_CONSTANT = 1,
    RK_ROLE_OUTPUT = 2,
    RK_ROLE_SCRATCH = 3,
    RK_ROLE_INPUT_SLICE = 4,
    RK_ROLE_WEIGHT = 5,
    RK_ROLE_ACCUMULATOR = 6,
    RK_ROLE_GATE = 7,
    RK_ROLE_IDENTITY = 8,
    RK_ROLE_STATE = 9
};

enum rk_dtype { RK_DTYPE_F16 = 0 };
enum rk_layout { RK_LAYOUT_CUBE_F16 = 1, RK_LAYOUT_WEIGHT_F16 = 2,
                 RK_LAYOUT_LINEAR_F16 = 3 };
enum rk_action_kind {
    RK_ACTION_HOST_PACK = 1,
    RK_ACTION_HOST_EXPAND = 2,
    RK_ACTION_HOST_CHECK = 3,
    RK_ACTION_UPLOAD = 4,
    RK_ACTION_NPU_TASK = 5,
    RK_ACTION_DOWNLOAD = 6,
    RK_ACTION_HOST_COMPUTE = 7,
    RK_ACTION_COMMIT = 8
};
enum rk_template_id {
    RK_TEMPLATE_MM_F16 = 1,
    RK_TEMPLATE_MM_ACC_F16 = 2,
    RK_TEMPLATE_EW_ADD_F16 = 3,
    RK_TEMPLATE_EW_MUL_F16 = 4,
    RK_TEMPLATE_LUT_SIGMOID_F16 = 5,
    RK_TEMPLATE_LUT_HARDSIGMOID_F16 = 6,
    RK_TEMPLATE_LUT_GELU_GATE_F16 = 7
};
enum rk_act_kind {
    RK_ACT_NONE = 0,
    RK_ACT_SIGMOID = 1,
    RK_ACT_HARDSIGMOID = 2,
    RK_ACT_GELU_GATE = 3
};
enum rk_submit_mode { RK_SUBMIT_CHAIN = 0, RK_SUBMIT_SEPARATE = 1 };
enum rk_core_policy { RK_CORE_SINGLE = 0, RK_CORE_AUTO = 1 };

/* Action flags. */
#define RK_ACTION_FLAG_TRANSPOSE_SOURCE 1u
#define RK_ACTION_FLAG_CACHE_CONSTANT 2u
/* Upload-only expansion of an explicit constant source: scalar fills every
 * logical destination element; channel requires rank-1 [N] == destination cols. */
#define RK_ACTION_FLAG_BROADCAST_SCALAR 4u
#define RK_ACTION_FLAG_BROADCAST_CHANNEL 8u

typedef struct rk_diag {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t code;
    int32_t native_status;
    uint32_t action_id;
    uint32_t has_action;
    char stage[RK_DIAG_STAGE_LEN];
    char message[RK_DIAG_MESSAGE_LEN];
    char source[RK_DIAG_SOURCE_LEN];
    char profile[RK_DIAG_PROFILE_LEN];
} rk_diag;

typedef struct rk_value_desc {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t id;
    uint32_t role;
    uint32_t dtype;
    uint32_t rank;
    uint32_t shape[2];
    uint64_t nbytes;
    const char *name;
    uint32_t name_len;
    uint32_t reserved;
} rk_value_desc;

typedef struct rk_tensor_desc {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t dtype;
    uint32_t layout;
    uint32_t rows;
    uint32_t cols;
    uint32_t physical_rows;
    uint32_t physical_cols;
    uint32_t atom;
    uint32_t reserved;
    uint64_t bytes;
} rk_tensor_desc;

typedef struct rk_buffer_desc {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t id;
    uint32_t role;
    uint32_t slot_id;
    uint32_t alignment;
    uint32_t layout_id;
    uint32_t padding_state;
    uint32_t lifetime_first;
    uint32_t lifetime_last;
    uint64_t offset;
    uint64_t capacity;
} rk_buffer_desc;

typedef struct rk_view_desc {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t id;
    uint32_t buffer_id;
    uint64_t offset_bytes;
    uint32_t valid_region[4];
    rk_tensor_desc tensor;
} rk_view_desc;

typedef struct rk_action_desc {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t id;
    uint32_t kind;
    uint32_t template_id;
    uint32_t ninputs;
    uint32_t noutputs;
    uint32_t ndeps;
    uint32_t source_binding;
    uint32_t source_row_offset;
    uint32_t source_col_offset;
    uint32_t source_row_stride;
    uint32_t flags;
    uint32_t act_kind;
    const uint32_t *inputs;
    const uint32_t *outputs;
    const uint32_t *deps;
    char origin[RK_ORIGIN_LEN];
} rk_action_desc;

typedef struct rk_submit_desc {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t id;
    uint32_t mode;
    uint32_t core_policy;
    uint32_t ntask_ids;
    uint32_t ndeps;
    const uint32_t *task_ids;
    const uint32_t *deps;
} rk_submit_desc;

/* One written tile of an output binding.  The view supplies the physical
 * packed tile; dst_row_offset/dst_col_offset place its logical rows/cols into
 * the row-major [rows][cols] output binding.  A binding is published by a set
 * of non-overlapping tiles that exactly cover it. */
typedef struct rk_publish_desc {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t binding_id;
    uint32_t view_id;
    uint32_t dst_row_offset;
    uint32_t dst_col_offset;
    uint32_t rows;
    uint32_t cols;
} rk_publish_desc;

typedef struct rk_plan_desc {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t schema_version;
    uint32_t flags;
    uint64_t profile_hash;
    uint64_t source_hash;
    const rk_value_desc *values;
    uint32_t nvalues;
    uint32_t reserved0;
    const rk_buffer_desc *buffers;
    uint32_t nbuffers;
    uint32_t reserved1;
    const rk_view_desc *views;
    uint32_t nviews;
    uint32_t reserved2;
    const rk_action_desc *actions;
    uint32_t nactions;
    uint32_t reserved3;
    const rk_submit_desc *submits;
    uint32_t nsubmits;
    uint32_t reserved4;
    const rk_publish_desc *publish;
    uint32_t npublish;
    uint32_t reserved5;
    uint64_t device_budget_bytes;
    uint64_t host_budget_bytes;
    uint64_t claimed_device_bytes;
    uint64_t claimed_host_bytes;
    /* Must equal RK_ABI_VERSION of the loaded library.  Catches a stale
     * librknpu_plan.so paired with a newer Python descriptor builder. */
    uint32_t runtime_abi_version;
    uint32_t reserved6;
} rk_plan_desc;

typedef struct rk_binding {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t id;
    uint32_t dtype;
    uint32_t rank;
    uint32_t shape[2];
    const void *data;
    uint64_t nbytes;
} rk_binding;

typedef struct rk_run_report {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t task_count;
    uint32_t submit_count;
    uint32_t fail_action_id;
    uint32_t numeric_mode;
    uint64_t upload_calls;
    uint64_t upload_bytes;
    uint64_t download_calls;
    uint64_t download_bytes;
    double pack_us;
    double emit_us;
    double submit_wait_us;
    double output_sync_us;
} rk_run_report;

typedef struct rk_plan rk_plan;
typedef struct rk_workspace rk_workspace;

uint32_t rk_abi_version(void);
/* Production target identity check.  The RK3588 build verifies the board
 * compatible string and kernel driver version before a workspace is created;
 * the test-only mock build intentionally supplies only the weak no-op. */
int rk_target_check(rk_diag *diag);
/* Pure parsing helper used by host regression tests: exact NUL-token compatible
 * match plus exact driver-version match. */
int rk_target_validate_text(const char *compat, size_t compat_len,
                            const char *version, rk_diag *diag);

int rk_plan_create(const rk_plan_desc *desc, rk_plan **out, rk_diag *diag);
void rk_plan_destroy(rk_plan *plan);
int rk_workspace_create(int fd, const rk_plan *plan,
                        const rk_binding *constants, uint32_t nconstants,
                        rk_workspace **out, rk_diag *diag);
int rk_workspace_run(rk_workspace *ws,
                     const rk_binding *inputs, uint32_t ninputs,
                     const rk_binding *outputs, uint32_t noutputs,
                     rk_run_report *report, rk_diag *diag);
int rk_workspace_destroy(rk_workspace *ws, rk_diag *diag);

#ifdef __cplusplus
}
#endif
#endif
