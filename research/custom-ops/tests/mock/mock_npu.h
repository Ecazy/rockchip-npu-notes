/* Test-only mock seam for the RKNPU host prototype.
 *
 * NOT part of the production ABI: nothing under src/ or include/ includes this
 * header, and librknpu_plan (the production library) never links the mock.
 *
 * Every declaration here is mirrored by an explicit ctypes signature in
 * python/rknpu_backend/runtime/ffi.py::LibRKPlan._MOCK_SIGNATURES, and the mirror
 * is checked at compile time by tests/mock/mock_signature_probe.c.
 */
#ifndef RK_MOCK_NPU_H
#define RK_MOCK_NPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Counters and controls. */
unsigned rk_mock_execs(void);
unsigned rk_mock_allocs(void);
int rk_mock_live_bos(void);
size_t rk_mock_resident(void);
unsigned rk_mock_tag_count(void);
unsigned rk_mock_tag_capacity(void);
unsigned rk_mock_tag_overflow(void);
unsigned rk_mock_untracked(void);
unsigned rk_mock_bo_count(void);
void rk_mock_fail_at(unsigned index);
void rk_mock_fail_alloc(unsigned index);
void rk_mock_inject_nan(unsigned on);
void rk_mock_corrupt_guard(unsigned mode);
void rk_mock_corrupt_guard_at(unsigned exec_index);
void rk_mock_delay_us(unsigned us);
void rk_mock_fail_prep_at(unsigned index);
unsigned rk_mock_prep_calls(void);
void rk_mock_force_tag_overflow(unsigned on);
void rk_mock_reset(void);

/* Introspection used by the BO / DMA self-checks. */
int rk_mock_dma_range(unsigned index, uint32_t *handle, uint32_t *dma,
                      size_t *size);
uint32_t rk_mock_dma_of(uint32_t handle);
uint64_t rk_mock_bo_dma_address(uint32_t handle);
int rk_mock_find_half(uint16_t pattern, unsigned *bo_index, size_t *offset);

#ifdef __cplusplus
}
#endif

#endif /* RK_MOCK_NPU_H */
