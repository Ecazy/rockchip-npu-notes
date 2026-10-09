#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <link.h>

static void *(*real_mmap)(void *, size_t, int, int, int, off_t);

struct rec { void *addr; size_t len; int fd; };
static struct rec recs[64];
static int nrecs;

static void dump_all(void) {
    char path[256];
    for (int i = 0; i < nrecs; i++) {
        char name[256];
        snprintf(name, sizeof(name), "/mnt/ssd/workspace/bn_lower_probe/bodump_%02d_fd%d.bin", i, recs[i].fd);
        FILE *f = fopen(name, "wb");
        if (!f) continue;
        fwrite(recs[i].addr, 1, recs[i].len, f);
        fclose(f);
        fprintf(stderr, "BODUMP %s addr=%p len=%zu\n", name, recs[i].addr, recs[i].len);
    }
}

void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
    if (!real_mmap) real_mmap = dlsym(RTLD_NEXT, "mmap");
    void *ret = real_mmap(addr, length, prot, flags, fd, offset);
    if (ret != MAP_FAILED && (flags & MAP_SHARED) && fd >= 3 && length >= 4096 && length <= (1u << 22)) {
        if (nrecs < 64) {
            recs[nrecs].addr = ret; recs[nrecs].len = length; recs[nrecs].fd = fd;
            nrecs++;
        }
        if (nrecs == 1) atexit(dump_all);
    }
    return ret;
}
