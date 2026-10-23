// SPDX-License-Identifier: GPL-3.0-or-later
/* Production RK3588 target identity gate.
 *
 * The test-only mock library deliberately does not link this file; the runtime
 * object's weak rk_target_check() no-op is used there instead.
 */
#include "rk_plan.h"
#include "rk_plan_internal.h"

#include <stdio.h>
#include <string.h>

#ifndef RK_TARGET_CHIP_COMPAT
#define RK_TARGET_CHIP_COMPAT "rockchip,rk3588"
#endif
#ifndef RK_TARGET_DRIVER_EXACT
#define RK_TARGET_DRIVER_EXACT "RKNPU driver: v0.9.8"
#endif

static int read_file_bytes(const char *path, char *out, size_t cap,
                           size_t *out_len) {
    if (!path || !out || !out_len || cap < 2u) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    size_t n = fread(out, 1u, cap - 1u, fp);
    int err = ferror(fp);
    fclose(fp);
    if (err) return -1;
    while (n && (out[n - 1u] == '\n' || out[n - 1u] == '\r' ||
                 out[n - 1u] == 0))
        n--;
    out[n] = 0;
    *out_len = n;
    return 0;
}

/* Device-tree compatible is a NUL-separated list.  Match one complete item,
 * never a substring: "rockchip,rk3588-extra" is not "rockchip,rk3588". */
static int compat_has_token(const char *buf, size_t len, const char *token) {
    if (!buf || !token) return 0;
    size_t token_len = strlen(token);
    size_t pos = 0;
    while (pos < len) {
        size_t end = pos;
        while (end < len && buf[end] != 0) end++;
        if (end - pos == token_len && !memcmp(buf + pos, token, token_len))
            return 1;
        if (end == len) break;
        pos = end + 1u;
    }
    return 0;
}

int rk_target_validate_text(const char *compat, size_t compat_len,
                            const char *version, rk_diag *diag) {
    if (!compat || !version) {
        rk_diag_set(diag, RK_E_TARGET_MISMATCH, "target",
                    "target environment text is missing", "", UINT32_MAX, 0);
        return RK_E_TARGET_MISMATCH;
    }
    if (!compat_has_token(compat, compat_len, RK_TARGET_CHIP_COMPAT)) {
        rk_diag_set(diag, RK_E_TARGET_MISMATCH, "target",
                    "board compatible list does not contain the exact RK3588 "
                    "target token", "", UINT32_MAX, 0);
        return RK_E_TARGET_MISMATCH;
    }
    if (strcmp(version, RK_TARGET_DRIVER_EXACT) != 0) {
        rk_diag_set(diag, RK_E_TARGET_MISMATCH, "target",
                    "RKNPU driver version is not the exact locked target version",
                    "", UINT32_MAX, 0);
        return RK_E_TARGET_MISMATCH;
    }
    return 0;
}

int rk_target_check(rk_diag *diag) {
    char compat[4096];
    size_t compat_len = 0;
    char version[256];
    size_t version_len = 0;
    if (read_file_bytes("/proc/device-tree/compatible", compat, sizeof compat,
                        &compat_len)) {
        rk_diag_set(diag, RK_E_TARGET_MISMATCH, "target",
                    "cannot read /proc/device-tree/compatible", "", UINT32_MAX, 0);
        return RK_E_TARGET_MISMATCH;
    }
    if (read_file_bytes("/sys/kernel/debug/rknpu/version", version,
                        sizeof version, &version_len)) {
        rk_diag_set(diag, RK_E_TARGET_MISMATCH, "target",
                    "cannot read the RKNPU driver version", "", UINT32_MAX, 0);
        return RK_E_TARGET_MISMATCH;
    }
    (void)version_len;
    return rk_target_validate_text(compat, compat_len, version, diag);
}
