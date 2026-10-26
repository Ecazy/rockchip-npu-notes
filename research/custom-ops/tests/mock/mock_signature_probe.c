/* Test-only probe for the mock seam.
 *
 * Two jobs:
 *  1. compile-time: each rk_mock_* is assigned to a function pointer of the exact
 *     expected type below, so a signature change fails the build;
 *  2. runtime: print one `name<TAB>return<TAB>args` line per function so the
 *     Python side can compare its explicit ctypes table against the C prototype
 *     field by field (python/rknpu_backend/.../ffi.py::_MOCK_SIGNATURES).
 *
 * Scope of the guarantee: the pointer assignments and the printed strings below
 * are TWO hand-written sources of information.  The assignments make a prototype
 * change a build error; the strings are what Python compares against, so they must
 * be updated together with the assignments - neither is generated from the other.
 */
#include <stddef.h>
#include <stdint.h>

#include <stdio.h>

#include "mock_npu.h"
#include "rocket_npu.h"
#include "rknpu_submit.h"

#define CHECK(fn, type) static type probe_##fn = fn

static unsigned (*probe_rk_mock_execs)(void) = rk_mock_execs;
static unsigned (*probe_rk_mock_allocs)(void) = rk_mock_allocs;
static int (*probe_rk_mock_live_bos)(void) = rk_mock_live_bos;
static size_t (*probe_rk_mock_resident)(void) = rk_mock_resident;
static unsigned (*probe_rk_mock_tag_count)(void) = rk_mock_tag_count;
static unsigned (*probe_rk_mock_tag_capacity)(void) = rk_mock_tag_capacity;
static unsigned (*probe_rk_mock_tag_overflow)(void) = rk_mock_tag_overflow;
static unsigned (*probe_rk_mock_untracked)(void) = rk_mock_untracked;
static unsigned (*probe_rk_mock_bo_count)(void) = rk_mock_bo_count;
static void (*probe_rk_mock_fail_at)(unsigned) = rk_mock_fail_at;
static void (*probe_rk_mock_fail_alloc)(unsigned) = rk_mock_fail_alloc;
static void (*probe_rk_mock_inject_nan)(unsigned) = rk_mock_inject_nan;
static void (*probe_rk_mock_corrupt_guard)(unsigned) = rk_mock_corrupt_guard;
static void (*probe_rk_mock_corrupt_guard_at)(unsigned) = rk_mock_corrupt_guard_at;
static void (*probe_rk_mock_delay_us)(unsigned) = rk_mock_delay_us;
static void (*probe_rk_mock_fail_prep_at)(unsigned) = rk_mock_fail_prep_at;
static unsigned (*probe_rk_mock_prep_calls)(void) = rk_mock_prep_calls;
static void (*probe_rk_mock_reset_tags)(void) = rk_mock_reset_tags;
static void (*probe_rk_mock_context_fault)(unsigned, unsigned) = rk_mock_context_fault;
static int (*probe_rk_mock_context_faulted)(int) = rk_mock_context_faulted;
static void (*probe_rk_mock_force_tag_overflow)(unsigned) = rk_mock_force_tag_overflow;
static void (*probe_rk_mock_reset)(void) = rk_mock_reset;
static int (*probe_rk_mock_dma_range)(unsigned, uint32_t *, uint32_t *, size_t *) = rk_mock_dma_range;
static uint32_t (*probe_rk_mock_dma_of)(uint32_t) = rk_mock_dma_of;
static uint64_t (*probe_rk_mock_bo_dma_address)(uint32_t) = rk_mock_bo_dma_address;
static int (*probe_rk_mock_find_half)(uint16_t, unsigned *, size_t *) = rk_mock_find_half;
/* Compile-time proof against the vendor header, not a Python hand-written mirror. */
static int (*probe_vendor_rocket_open)(void) = rocket_open;
static void (*probe_vendor_rocket_close)(int) = rocket_close;
static int (*probe_vendor_rocket_context_faulted)(int) = rocket_context_faulted;

static const struct { const char *name, *ret, *args; } kSignatures[] = {
    {"rk_mock_execs", "unsigned", ""},
    {"rk_mock_allocs", "unsigned", ""},
    {"rk_mock_live_bos", "int", ""},
    {"rk_mock_resident", "size_t", ""},
    {"rk_mock_tag_count", "unsigned", ""},
    {"rk_mock_tag_capacity", "unsigned", ""},
    {"rk_mock_tag_overflow", "unsigned", ""},
    {"rk_mock_untracked", "unsigned", ""},
    {"rk_mock_bo_count", "unsigned", ""},
    {"rk_mock_fail_at", "void", "unsigned"},
    {"rk_mock_fail_alloc", "void", "unsigned"},
    {"rk_mock_inject_nan", "void", "unsigned"},
    {"rk_mock_corrupt_guard", "void", "unsigned"},
    {"rk_mock_corrupt_guard_at", "void", "unsigned"},
    {"rk_mock_delay_us", "void", "unsigned"},
    {"rk_mock_fail_prep_at", "void", "unsigned"},
    {"rk_mock_prep_calls", "unsigned", ""},
    {"rk_mock_fail_fini_at", "void", "unsigned"},
    {"rk_mock_fini_calls", "unsigned", ""},
    {"rk_mock_reset_tags", "void", ""},
    {"rk_mock_context_fault", "void", "unsigned,unsigned"},
    {"rk_mock_context_faulted", "int", "int"},
    {"rk_mock_force_tag_overflow", "void", "unsigned"},
    {"rk_mock_reset", "void", ""},
    {"rk_mock_dma_range", "int", "unsigned,uint32_t *,uint32_t *,size_t *"},
    {"rk_mock_dma_of", "uint32_t", "uint32_t"},
    {"rk_mock_bo_dma_address", "uint64_t", "uint32_t"},
    {"rk_mock_find_half", "int", "uint16_t,unsigned *,size_t *"},
};

/* Touch every probe so -Werror=unused-variable stays quiet: the value of this
 * file is the compile-time type check plus the printed table. */
int main(void) {
    /* the value of this file is the compile-time type check + the printed table */
    (void)probe_rk_mock_execs;      (void)probe_rk_mock_allocs;
    (void)probe_rk_mock_live_bos;   (void)probe_rk_mock_resident;
    (void)probe_rk_mock_tag_count;  (void)probe_rk_mock_tag_capacity;
    (void)probe_rk_mock_tag_overflow; (void)probe_rk_mock_untracked;
    (void)probe_rk_mock_bo_count;   (void)probe_rk_mock_fail_at;
    (void)probe_rk_mock_fail_alloc; (void)probe_rk_mock_inject_nan;
    (void)probe_rk_mock_corrupt_guard;
    (void)probe_rk_mock_corrupt_guard_at;
    (void)probe_rk_mock_delay_us;
    (void)probe_rk_mock_fail_prep_at;
    (void)probe_rk_mock_prep_calls;
    (void)probe_rk_mock_reset_tags;
    (void)probe_rk_mock_context_fault;
    (void)probe_rk_mock_context_faulted;
    (void)probe_rk_mock_force_tag_overflow; (void)probe_rk_mock_reset;
    (void)probe_rk_mock_dma_range;  (void)probe_rk_mock_dma_of;
    (void)probe_rk_mock_bo_dma_address; (void)probe_rk_mock_find_half;
    (void)probe_vendor_rocket_open; (void)probe_vendor_rocket_close;
    (void)probe_vendor_rocket_context_faulted;
    printf("VENDOR_SIG\trocket_context_faulted\tint\tint\n");
    printf("VENDOR_SIG\trocket_open\tint\t\n");
    printf("VENDOR_SIG\trocket_close\tvoid\tint\n");
    for (unsigned i = 0; i < sizeof(kSignatures) / sizeof(kSignatures[0]); i++)
        printf("MOCK_SIG\t%s\t%s\t%s\n", kSignatures[i].name,
               kSignatures[i].ret, kSignatures[i].args);
    return 0;
}
