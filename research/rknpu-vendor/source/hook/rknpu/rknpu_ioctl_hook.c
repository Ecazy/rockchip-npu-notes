// Copyright (c) 2026 eperf-exporter contributors.
// SPDX-License-Identifier: Apache-2.0

#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <fcntl.h>
#include <linux/ioctl.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

struct rknpu_subcore_task {
  uint32_t task_start;
  uint32_t task_number;
};

struct rknpu_submit {
  uint32_t flags;
  uint32_t timeout;
  uint32_t task_start;
  uint32_t task_number;
  uint32_t task_counter;
  int32_t priority;
  uint64_t task_obj_addr;
  uint32_t iommu_domain_id;
  uint32_t reserved;
  uint64_t task_base_addr;
  int64_t hw_elapse_time;
  uint32_t core_mask;
  int32_t fence_fd;
  struct rknpu_subcore_task subcore_task[5];
};

struct rknpu_mem_create {
  uint32_t handle;
  uint32_t flags;
  uint64_t size;
  uint64_t obj_addr;
  uint64_t dma_addr;
  uint64_t sram_size;
  int32_t iommu_domain_id;
  uint32_t core_mask;
};

struct rknpu_mem_map {
  uint32_t handle;
  uint32_t reserved;
  uint64_t offset;
};

struct rknpu_mem_sync {
  uint32_t flags;
  uint32_t reserved;
  uint64_t obj_addr;
  uint64_t offset;
  uint64_t size;
};

struct rknpu_mem_destroy {
  uint32_t handle;
  uint32_t reserved;
  uint64_t obj_addr;
};

struct rknpu_task {
  uint32_t flags;
  uint32_t op_idx;
  uint32_t enable_mask;
  uint32_t int_mask;
  uint32_t int_clear;
  uint32_t int_status;
  uint32_t regcfg_amount;
  uint32_t regcfg_offset;
  uint64_t regcmd_addr;
} __attribute__((packed));

typedef int (*ioctl_fn)(int, unsigned long, ...);
typedef void* (*mmap_fn)(void*, size_t, int, int, int, off_t);
typedef void* (*mmap64_fn)(void*, size_t, int, int, int, off64_t);

struct memory_record {
  int fd;
  uint32_t handle;
  uint64_t size;
  uint64_t obj_addr;
  uint64_t dma_addr;
  uint64_t offset;
  void* mapped_address;
  size_t mapped_length;
  int32_t iommu_domain_id;
};

#define MAX_MEMORY_RECORDS 4096
#define MAX_REGCMD_BYTES 16384
#define REGCMD_HEX_BUDGET 32768

struct rknpu_task_capture {
  uint32_t op_idx;
  uint32_t flags;
  uint32_t enable_mask;
  uint32_t int_mask;
  uint32_t int_clear;
  uint32_t int_status;
  uint32_t regcfg_amount;
  uint32_t regcfg_offset;
  uint64_t regcmd_addr;
};

enum trace_stage {
  TRACE_STAGE_STARTUP = 0,
  TRACE_STAGE_LANGUAGE_INIT,
  TRACE_STAGE_VISION_INIT,
  TRACE_STAGE_PREPROCESS,
  TRACE_STAGE_VISION,
  TRACE_STAGE_LANGUAGE_PREFILL,
  TRACE_STAGE_LANGUAGE_DECODE,
  TRACE_STAGE_IDLE,
};

static void write_record(const char* record, size_t length);

static ioctl_fn real_ioctl;
static mmap_fn real_mmap;
static mmap64_fn real_mmap64;
static int trace_fd = -1;
static volatile int current_stage = TRACE_STAGE_STARTUP;
static volatile unsigned long long sequence_number;
/* Native prefix probe: truncate a matching submit's effective subcore range to
 * EPERF_RKNPU_PROBE_L tasks, forward to the driver, record hw_elapse_time.
 * driver invariant: task_number == sum(subcore_task[i].task_number); for a
 * core_mask=0 auto submit the runtime sets subcore_task[0..2] identical and
 * task_number = 3 x per-core. */
static int g_probe_enabled;
static int g_probe_L = 1;
static int g_probe_min_eff = 4000;
static int g_probe_exit;
static int g_probe_nth = 1;
static int g_probe_op_idx = -1;      /* EPERF_RKNPU_PROBE_OP_IDX: require tasks op_idx */
static int g_probe_require_all;      /* EPERF_RKNPU_PROBE_REQUIRE_ALL_TASKS_SAME_OP */
static int g_probe_start;            /* EPERF_RKNPU_PROBE_START: task_start offset */
static int g_probe_neutralize;       /* EPERF_RKNPU_PROBE_NEUTRALIZE: null the removed tail */
static int g_probe_timeout_ms;       /* EPERF_RKNPU_PROBE_TIMEOUT_MS: fail-stop cap on submit */
static int g_probe_reload_l;         /* EPERF_RKNPU_PROBE_RELOAD_L: re-read L env for each run */
static int g_probe_seen;
static int g_probe_saved_tn;
static struct rknpu_subcore_task g_probe_saved_sc[5];
static int g_probe_done;
static int g_probe_orig_tn;
static void* g_probe_arg;            /* identifies the truncated ioctl (multi-thread safe) */
static uint32_t g_probe_newsum;      /* truncated total task_number, for identity check */

/* Post-submit memory dumps for intermediate-tensor validation.  Each spec is
 * "ioaddr:size:path"; up to 8 ranges can be captured in one run. */
static int resolve_regcmd_pointer(uint64_t address, uint32_t iommu_domain_id,
                                  const void** host, size_t* available,
                                  const char** via);

struct dump_spec {
  uint64_t addr;
  uint64_t size;
  char path[256];
};
static struct dump_spec g_dump_specs[8];
static int g_dump_n;
static int g_dump_configured;
static int g_dump_probe_only;

static void configure_dumps(void) {
  if (g_dump_configured) return;
  g_dump_configured = 1;
  const char* mode = getenv("EPERF_RKNPU_DUMP_MODE");
  g_dump_probe_only = mode != NULL && strcmp(mode, "probe") == 0;
  for (int i = 0; i < 8; ++i) {
    char name[32];
    snprintf(name, sizeof(name), "EPERF_RKNPU_DUMP%d", i);
    const char* spec = getenv(name);
    if (spec == NULL || spec[0] == '\0') continue;
    char buf[512];
    snprintf(buf, sizeof(buf), "%s", spec);
    char* p1 = strchr(buf, ':');
    if (p1 == NULL) continue;
    *p1 = '\0';
    char* p2 = strchr(p1 + 1, ':');
    if (p2 == NULL) continue;
    *p2 = '\0';
    struct dump_spec* d = &g_dump_specs[g_dump_n];
    d->addr = strtoull(buf, NULL, 0);
    d->size = strtoull(p1 + 1, NULL, 0);
    snprintf(d->path, sizeof(d->path), "%s", p2 + 1);
    if (d->size != 0 && d->path[0] != '\0') g_dump_n++;
  }
}

static void dump_memory_ranges(const struct rknpu_submit* s, int result,
                               int is_probe) {
  if (s == NULL || result < 0) return;
  configure_dumps();
  if (g_dump_n == 0) return;
  if (g_dump_probe_only && !is_probe) return;
  for (int i = 0; i < g_dump_n; ++i) {
    const void* host = NULL;
    size_t available = 0;
    const char* via = "none";
    if (!resolve_regcmd_pointer(g_dump_specs[i].addr, s->iommu_domain_id,
                                &host, &available, &via) ||
        host == NULL || available == 0) {
      fprintf(stderr, "DUMP_MISS addr=0x%llx size=0x%llx path=%s\n",
              (unsigned long long)g_dump_specs[i].addr,
              (unsigned long long)g_dump_specs[i].size, g_dump_specs[i].path);
      continue;
    }
    size_t n = (size_t)g_dump_specs[i].size;
    if (n > available) n = available;
    FILE* f = fopen(g_dump_specs[i].path, "wb");
    if (f == NULL) {
      fprintf(stderr, "DUMP_OPEN_FAIL path=%s\n", g_dump_specs[i].path);
      continue;
    }
    size_t written = fwrite(host, 1, n, f);
    fclose(f);
    fprintf(stderr,
            "DUMP addr=0x%llx size_written=0x%zx requested=0x%llx via=%s path=%s\n",
            (unsigned long long)g_dump_specs[i].addr, written,
            (unsigned long long)g_dump_specs[i].size, via,
            g_dump_specs[i].path);
  }
}
static __thread int in_hook;
static volatile int memory_lock;
static struct memory_record memory_records[MAX_MEMORY_RECORDS];
static size_t regcmd_max_bytes = MAX_REGCMD_BYTES;

static void lock_memory_records(void) {
  while (__sync_lock_test_and_set(&memory_lock, 1) != 0) {
    syscall(SYS_sched_yield);
  }
}

static void unlock_memory_records(void) {
  __sync_lock_release(&memory_lock);
}

static int request_matches(unsigned long request, unsigned int command,
                           size_t size) {
  const unsigned int type = (unsigned int)_IOC_TYPE(request);
  const unsigned int number = (unsigned int)_IOC_NR(request);
  return (type == (unsigned int)'d' && number == 0x40U + command &&
          _IOC_SIZE(request) == size) ||
         (type == (unsigned int)'r' && number == command &&
          _IOC_SIZE(request) == size);
}

static struct memory_record* find_record_by_handle(int fd, uint32_t handle) {
  for (size_t index = 0; index < MAX_MEMORY_RECORDS; ++index) {
    struct memory_record* record = &memory_records[index];
    if (record->obj_addr != 0 && record->fd == fd &&
        record->handle == handle) {
      return record;
    }
  }
  return NULL;
}

// Match by obj_addr, preferring the record from the caller's IOMMU domain.
// The same IOVA range can exist in more than one domain (the rkllm stack uses
// domain 1 while plain RKNN uses 0), so a domain-blind match can return the
// wrong buffer.  Falls back to any domain only when nothing matches.
static struct memory_record* find_record_by_object(uint64_t obj_addr,
                                                   uint32_t iommu_domain_id) {
  struct memory_record* fallback = NULL;
  for (size_t index = 0; index < MAX_MEMORY_RECORDS; ++index) {
    struct memory_record* record = &memory_records[index];
    if (record->obj_addr != obj_addr) {
      continue;
    }
    if (record->iommu_domain_id == (int32_t)iommu_domain_id) {
      return record;
    }
    if (fallback == NULL) {
      fallback = record;
    }
  }
  return fallback;
}

static void track_mem_create(int fd, const struct rknpu_mem_create* create) {
  if (create == NULL || create->obj_addr == 0) {
    return;
  }
  lock_memory_records();
  struct memory_record* record = find_record_by_handle(fd, create->handle);
  if (record == NULL) {
    for (size_t index = 0; index < MAX_MEMORY_RECORDS; ++index) {
      if (memory_records[index].obj_addr == 0) {
        record = &memory_records[index];
        break;
      }
    }
  }
  if (record != NULL) {
    record->fd = fd;
    record->handle = create->handle;
    record->size = create->size;
    record->obj_addr = create->obj_addr;
    record->dma_addr = create->dma_addr;
    record->iommu_domain_id = create->iommu_domain_id;
  }
  unlock_memory_records();
}

static void track_mem_map(int fd, const struct rknpu_mem_map* map) {
  if (map == NULL || map->offset == 0) {
    return;
  }
  lock_memory_records();
  struct memory_record* record = find_record_by_handle(fd, map->handle);
  if (record != NULL) {
    record->offset = map->offset;
  }
  unlock_memory_records();
}

static void track_mem_destroy(int fd,
                              const struct rknpu_mem_destroy* destroy) {
  if (destroy == NULL) {
    return;
  }
  lock_memory_records();
  struct memory_record* record = find_record_by_handle(fd, destroy->handle);
  if (record != NULL &&
      (destroy->obj_addr == 0 || record->obj_addr == destroy->obj_addr)) {
    memset(record, 0, sizeof(*record));
  }
  unlock_memory_records();
}

static uint64_t track_mmap_result(int fd, uint64_t offset, void* address,
                                  size_t length) {
  if (address == MAP_FAILED || address == NULL) {
    return 0;
  }
  uint64_t matched_object = 0;
  lock_memory_records();
  for (size_t index = 0; index < MAX_MEMORY_RECORDS; ++index) {
    struct memory_record* record = &memory_records[index];
    if (record->obj_addr != 0 && record->fd == fd &&
        record->offset == offset) {
      record->mapped_address = address;
      record->mapped_length = length;
      matched_object = record->obj_addr;
      break;
    }
  }
  unlock_memory_records();
  return matched_object;
}

static size_t capture_task_details(const struct rknpu_submit* submit,
                                   struct rknpu_task_capture* captures,
                                   size_t capacity) {
  if (submit == NULL || captures == NULL || capacity == 0) {
    return 0;
  }
  size_t captured = 0;
  lock_memory_records();
  struct memory_record* record =
      find_record_by_object(submit->task_obj_addr, submit->iommu_domain_id);
  if (record != NULL && record->mapped_address != NULL) {
    const size_t task_count =
        submit->task_number < capacity ? submit->task_number : capacity;
    const size_t start = submit->task_start;
    const size_t required = (start + task_count) * sizeof(struct rknpu_task);
    const size_t available = record->mapped_length != 0
                                 ? record->mapped_length
                                 : (size_t)record->size;
    if (required <= available) {
      const struct rknpu_task* tasks =
          (const struct rknpu_task*)record->mapped_address;
      for (size_t index = 0; index < task_count; ++index) {
        const struct rknpu_task* task = &tasks[start + index];
        captures[index].op_idx = task->op_idx;
        captures[index].flags = task->flags;
        captures[index].enable_mask = task->enable_mask;
        captures[index].int_mask = task->int_mask;
        captures[index].int_clear = task->int_clear;
        captures[index].int_status = task->int_status;
        captures[index].regcfg_amount = task->regcfg_amount;
        captures[index].regcfg_offset = task->regcfg_offset;
        captures[index].regcmd_addr = task->regcmd_addr;
      }
      captured = task_count;
    }
  }
  unlock_memory_records();
  return captured;
}

// Best-effort user-space translation of an NPU command address (IOVA) to a
// mapped host buffer.  The runtime allocates the regcmd BO with
// RKNPU_MEM_CREATE and maps it with mmap; we match against the mmap offset,
// obj_addr and dma_addr ranges recorded earlier.
static struct memory_record* g_last_resolved_record = NULL;
static int resolve_regcmd_pointer(uint64_t address, uint32_t iommu_domain_id,
                                  const void** host, size_t* available,
                                  const char** via) {
  int found = 0;
  g_last_resolved_record = NULL;
  *host = NULL;
  *available = 0;
  *via = "none";
  // Two passes: first only records from the caller's IOMMU domain, then any
  // (fallback) so a missing/zero domain still resolves as before.
  for (int pass = 0; pass < 2 && !found; ++pass) {
  lock_memory_records();
  for (size_t index = 0; index < MAX_MEMORY_RECORDS; ++index) {
    struct memory_record* record = &memory_records[index];
    if (record->obj_addr == 0 || record->mapped_address == NULL) {
      continue;
    }
    if (pass == 0 && record->iommu_domain_id != (int32_t)iommu_domain_id) {
      continue;
    }
    const size_t span = record->mapped_length != 0
                            ? record->mapped_length
                            : (size_t)record->size;
    if (span == 0) {
      continue;
    }
    uint64_t base = 0;
    // Some runtimes (observed on the RKLLM language path) hand the driver a
    // sign-extended 64-bit command IOVA (0xffffffffXXXXXXXX) while the BO was
    // recorded with a 32-bit dma_addr.  Match the low 32 bits as a fallback.
    uint64_t address32 = address & 0xffffffffULL;
    uint64_t effective = address;
    const char* matched_via = NULL;
    if (record->offset != 0 && address >= record->offset &&
        address < record->offset + span) {
      base = record->offset;
      matched_via = "mmap_offset";
    } else if (address >= record->obj_addr &&
               address < record->obj_addr + record->size) {
      base = record->obj_addr;
      matched_via = "obj_addr";
    } else if (record->dma_addr != 0 && address >= record->dma_addr &&
               address < record->dma_addr + record->size) {
      base = record->dma_addr;
      matched_via = "dma_addr";
    } else if (record->dma_addr != 0 && address32 >= record->dma_addr &&
               address32 < record->dma_addr + record->size) {
      base = record->dma_addr;
      effective = address32;
      matched_via = "dma_addr32";
    } else {
      continue;
    }
    *host = (const char*)record->mapped_address + (effective - base);
    *available = span - (size_t)(effective - base);
    *via = matched_via;
    g_last_resolved_record = record;
    found = 1;
    break;
  }
  unlock_memory_records();
  }
  return found;
}

static void write_task_record(unsigned long long sequence, size_t task_index,
                              const struct rknpu_task_capture* capture,
                              uint64_t task_base_addr, uint32_t iommu_domain_id,
                              int resolved, const char* via) {
  if (trace_fd < 0 || capture == NULL) {
    return;
  }
  char record[768];
  int length = snprintf(
      record, sizeof(record),
      "{\"schema_version\":1,\"event\":\"task\",\"sequence\":%llu,"
      "\"task_index\":%zu,\"op_idx\":%u,\"flags\":%u,"
      "\"enable_mask\":%u,\"int_mask\":%u,\"int_clear\":%u,\"int_status\":%u,"
      "\"regcfg_amount\":%u,\"regcfg_offset\":%u,"
      "\"regcmd_addr\":\"0x%llx\",\"resolved\":%s,\"resolved_via\":\"%s\","
      "\"task_base_addr\":\"0x%llx\",\"iommu_domain_id\":%u}\n",
      sequence, task_index, capture->op_idx, capture->flags,
      capture->enable_mask, capture->int_mask, capture->int_clear,
      capture->int_status,
      capture->regcfg_amount, capture->regcfg_offset,
      (unsigned long long)capture->regcmd_addr, resolved ? "true" : "false",
      via, (unsigned long long)task_base_addr, iommu_domain_id);
  if (length > 0 && (size_t)length < sizeof(record)) {
    write_record(record, (size_t)length);
  }
}

static void write_regcmd_record(unsigned long long sequence, size_t task_index,
                                const struct rknpu_task_capture* capture,
                                const void* host, size_t available) {
  if (trace_fd < 0 || capture == NULL || host == NULL || available == 0) {
    return;
  }
  const size_t limit = regcmd_max_bytes != 0 ? regcmd_max_bytes : MAX_REGCMD_BYTES;
  // The driver treats regcfg_amount as a count of 64-bit regcfg/NPUOP entries
  // and additionally fetches 4 PC trailer entries, padded up to an even count.
  size_t entries = (size_t)capture->regcfg_amount + 4u;
  entries = (entries + 1u) & ~(size_t)1u;
  const size_t expected_bytes = entries * sizeof(uint64_t);
  // Never emit a short payload as if it were the whole stream: the record now
  // carries expected_bytes and an explicit truncated flag, so a 1024/1102
  // capture cannot be mistaken for a complete command stream downstream.
  size_t bytes = expected_bytes;
  if (bytes == 0 || bytes > available) bytes = available;
  if (bytes > limit) bytes = limit;
  if (bytes > (size_t)(REGCMD_HEX_BUDGET / 2)) bytes = (size_t)(REGCMD_HEX_BUDGET / 2);
  const unsigned char* data = (const unsigned char*)host;
  char record[REGCMD_HEX_BUDGET + 512];
  int length = snprintf(
      record, sizeof(record),
      "{\"schema_version\":1,\"event\":\"regcmd\",\"sequence\":%llu,"
      "\"task_index\":%zu,\"op_idx\":%u,\"regcfg_amount\":%u,"
      "\"expected_bytes\":%zu,\"bytes\":%zu,\"truncated\":%s,\"hex\":\"",
      sequence, task_index, capture->op_idx, capture->regcfg_amount,
      expected_bytes, bytes, bytes < expected_bytes ? "true" : "false");
  if (length <= 0 || (size_t)length >= sizeof(record)) {
    return;
  }
  size_t used = (size_t)length;
  static const char digits[] = "0123456789abcdef";
  for (size_t index = 0; index < bytes; ++index) {
    if (used + 3 >= sizeof(record)) break;
    record[used++] = digits[(data[index] >> 4) & 0xf];
    record[used++] = digits[data[index] & 0xf];
  }
  if (used + 3 >= sizeof(record)) {
    return;
  }
  record[used++] = '"';
  record[used++] = '}';
  record[used++] = '\n';
  write_record(record, used);
}

static const char* stage_name(int stage) {
  switch (stage) {
    case TRACE_STAGE_LANGUAGE_INIT:
      return "language_init";
    case TRACE_STAGE_VISION_INIT:
      return "vision_init";
    case TRACE_STAGE_PREPROCESS:
      return "preprocess";
    case TRACE_STAGE_VISION:
      return "vision";
    case TRACE_STAGE_LANGUAGE_PREFILL:
      return "language_prefill";
    case TRACE_STAGE_LANGUAGE_DECODE:
      return "language_decode";
    case TRACE_STAGE_IDLE:
      return "idle";
    case TRACE_STAGE_STARTUP:
    default:
      return "startup";
  }
}

static int parse_stage(const char* stage) {
  if (stage == NULL) {
    return TRACE_STAGE_IDLE;
  }
  if (strcmp(stage, "startup") == 0) {
    return TRACE_STAGE_STARTUP;
  }
  if (strcmp(stage, "language_init") == 0) {
    return TRACE_STAGE_LANGUAGE_INIT;
  }
  if (strcmp(stage, "vision_init") == 0) {
    return TRACE_STAGE_VISION_INIT;
  }
  if (strcmp(stage, "preprocess") == 0) {
    return TRACE_STAGE_PREPROCESS;
  }
  if (strcmp(stage, "vision") == 0) {
    return TRACE_STAGE_VISION;
  }
  if (strcmp(stage, "language_prefill") == 0) {
    return TRACE_STAGE_LANGUAGE_PREFILL;
  }
  if (strcmp(stage, "language_decode") == 0) {
    return TRACE_STAGE_LANGUAGE_DECODE;
  }
  return TRACE_STAGE_IDLE;
}

static long long monotonic_raw_ns(void) {
  struct timespec timestamp;
  if (syscall(SYS_clock_gettime, CLOCK_MONOTONIC_RAW, &timestamp) != 0) {
    return 0;
  }
  return (long long)timestamp.tv_sec * 1000000000LL + timestamp.tv_nsec;
}

static void write_record(const char* record, size_t length) {
  if (trace_fd < 0 || record == NULL || length == 0) {
    return;
  }
  const char* cursor = record;
  size_t remaining = length;
  while (remaining > 0) {
    long written = syscall(SYS_write, trace_fd, cursor, remaining);
    if (written <= 0) {
      return;
    }
    cursor += (size_t)written;
    remaining -= (size_t)written;
  }
}

static void write_meta_record(void) {
  char record[512];
  int length = snprintf(
      record, sizeof(record),
      "{\"schema_version\":1,\"event\":\"meta\",\"hook\":"
      "\"rknpu_ioctl\",\"pid\":%ld,\"clock\":\"CLOCK_MONOTONIC_RAW\","
      "\"hw_elapse_time_unit\":\"ns\","
      "\"submit_task_op_idx_storage\":\"task_events\"}\n",
      (long)syscall(SYS_getpid));
  if (length > 0 && (size_t)length < sizeof(record)) {
    write_record(record, (size_t)length);
  }
}

static void write_mem_create_record(int fd,
                                    const struct rknpu_mem_create* create) {
  if (trace_fd < 0 || create == NULL) {
    return;
  }
  char record[768];
  int length = snprintf(
      record, sizeof(record),
      "{\"schema_version\":1,\"event\":\"mem_create\",\"fd\":%d,"
      "\"handle\":%u,\"flags\":%u,\"size\":%llu,"
      "\"obj_addr\":\"0x%llx\",\"dma_addr\":\"0x%llx\","
      "\"sram_size\":%llu,\"iommu_domain_id\":%d,\"core_mask\":%u}\n",
      fd, create->handle, create->flags, (unsigned long long)create->size,
      (unsigned long long)create->obj_addr,
      (unsigned long long)create->dma_addr,
      (unsigned long long)create->sram_size, create->iommu_domain_id,
      create->core_mask);
  if (length > 0 && (size_t)length < sizeof(record)) {
    write_record(record, (size_t)length);
  }
}

static void write_mem_map_record(int fd, const struct rknpu_mem_map* map) {
  if (trace_fd < 0 || map == NULL) {
    return;
  }
  char record[384];
  int length = snprintf(
      record, sizeof(record),
      "{\"schema_version\":1,\"event\":\"mem_map\",\"fd\":%d,"
      "\"handle\":%u,\"offset\":%llu}\n",
      fd, map->handle, (unsigned long long)map->offset);
  if (length > 0 && (size_t)length < sizeof(record)) {
    write_record(record, (size_t)length);
  }
}

static void write_mmap_record(int fd, uint64_t offset, void* address,
                              size_t length, uint64_t matched_object) {
  if (trace_fd < 0) {
    return;
  }
  char record[512];
  int record_length = snprintf(
      record, sizeof(record),
      "{\"schema_version\":1,\"event\":\"mmap\",\"fd\":%d,"
      "\"offset\":%llu,\"length\":%zu,\"address\":\"0x%llx\","
      "\"matched_object\":\"0x%llx\"}\n",
      fd, (unsigned long long)offset, length,
      (unsigned long long)(uintptr_t)address,
      (unsigned long long)matched_object);
  if (record_length > 0 && (size_t)record_length < sizeof(record)) {
    write_record(record, (size_t)record_length);
  }
}

__attribute__((constructor)) static void initialize_hook(void) {
  in_hook = 1;
  void* symbol = dlsym(RTLD_NEXT, "ioctl");
  _Static_assert(sizeof(real_ioctl) == sizeof(symbol),
                 "function and object pointers must have the same size");
  memcpy(&real_ioctl, &symbol, sizeof(real_ioctl));
  symbol = dlsym(RTLD_NEXT, "mmap");
  _Static_assert(sizeof(real_mmap) == sizeof(symbol),
                 "function and object pointers must have the same size");
  memcpy(&real_mmap, &symbol, sizeof(real_mmap));
  symbol = dlsym(RTLD_NEXT, "mmap64");
  _Static_assert(sizeof(real_mmap64) == sizeof(symbol),
                 "function and object pointers must have the same size");
  memcpy(&real_mmap64, &symbol, sizeof(real_mmap64));
  const char* regcmd_limit = getenv("EPERF_RKNPU_REGCMD_MAX_BYTES");
  if (regcmd_limit != NULL && regcmd_limit[0] != '\0') {
    long value = strtol(regcmd_limit, NULL, 10);
    if (value > 0) {
      regcmd_max_bytes = (size_t)value;
    }
  }
  const char* output = getenv("EPERF_RKNPU_IOCTL_TRACE");
  if (output != NULL && output[0] != '\0') {
    trace_fd = (int)syscall(SYS_openat, AT_FDCWD, output,
                            O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    write_meta_record();
  }
  in_hook = 0;
}

__attribute__((visibility("default"))) void eperf_rknpu_trace_set_stage(
    const char* stage) {
  int parsed = parse_stage(stage);
  __atomic_store_n(&current_stage, parsed, __ATOMIC_RELEASE);
  if (trace_fd >= 0) {
    char record[384];
    int length = snprintf(
        record, sizeof(record),
        "{\"schema_version\":1,\"event\":\"stage\",\"stage\":\"%s\","
        "\"timestamp_ns\":%lld,\"pid\":%ld,\"tid\":%ld}\n",
        stage_name(parsed), monotonic_raw_ns(), (long)syscall(SYS_getpid),
        (long)syscall(SYS_gettid));
    if (length > 0 && (size_t)length < sizeof(record)) {
      write_record(record, (size_t)length);
    }
  }
}

static int is_rknpu_submit(unsigned long request) {
  return request_matches(request, 0x01U, sizeof(struct rknpu_submit));
}

// Optional read of the driver's memory-traffic counters via RKNPU_ACTION
// (GET_DT_WR/DT_RD/WT_RD/TOTAL_RW = 14/15/16/17).  The counters are tracked per
// submitting context, so the query must use the same fd as the submit.  Gated
// by EPERF_RKNPU_RW_COUNTER to avoid overhead on normal captures.
struct rknpu_action_query {
  uint32_t flags;
  uint32_t value;
};

static int rw_counter_enabled(void) {
  static int cached = -1;
  if (cached < 0) {
    const char* env = getenv("EPERF_RKNPU_RW_COUNTER");
    cached = (env != NULL && env[0] != '\0' && env[0] != '0') ? 1 : 0;
  }
  return cached;
}

static unsigned int query_rw_amount(int fd, uint32_t action) {
  struct rknpu_action_query query;
  query.flags = action;
  query.value = 0;
  unsigned long request =
      _IOC(_IOC_READ | _IOC_WRITE, 'd', 0x40U, sizeof(query));
  int result = real_ioctl != NULL ? real_ioctl(fd, request, &query)
                                  : (int)syscall(SYS_ioctl, fd, request, &query);
  if (result != 0) {
    request = _IOC(_IOC_READ | _IOC_WRITE, 'r', 0x00U, sizeof(query));
    result = real_ioctl != NULL ? real_ioctl(fd, request, &query)
                                : (int)syscall(SYS_ioctl, fd, request, &query);
  }
  return result == 0 ? query.value : 0U;
}

/* In-process block watcher: sample NPU block S_STATUS from /dev/mem while the
 * main thread is blocked inside the real submit ioctl.
 *
 * SAFETY / BUDGET RULES (see research/scheduling/REGCMD-TIMING-OBSERVABILITY.md):
 *   - every MMIO read goes through bw_read_u32() and counts against
 *     EPERF_RKNPU_BLOCKWATCH_MAX_READS; a sample that runs out of budget is
 *     discarded instead of being published half-read;
 *   - edge mode (one sample per PC counter transition) uses the SAME pacer and
 *     the SAME budget as the free-running mode: the poll read is throttled by
 *     EPERF_RKNPU_BLOCKWATCH_POLL_NS, so "one record per transition" can no
 *     longer degenerate into an unbounded MMIO loop;
 *   - the phase delay is measured from bw_arm(), which the caller invokes
 *     immediately before entering the real ioctl, not from thread start;
 *   - a coarse power guard re-reads the debugfs power file during sampling and
 *     aborts the run if the domain drops;
 *   - every register read is bracketed by its own read_begin_ns/read_end_ns.
 * Sampler state is published with release/acquire atomics: ready -> armed ->
 * samples -> stopped, so a phase delay can never elapse before the submit.
 */
struct bw_core {
  /* Group bracket: one pair around ALL register reads of this core's snapshot
   * (1..5 reads), not one pair per register.  Per-register brackets would cost
   * two clock_gettime calls per register and dominate the read itself. */
  uint64_t gb, ge;
  uint32_t ts, cna, core, dpu, rdma;
};
struct bw_sample {
  uint64_t t;     /* sample attempt start (monotonic raw ns) */
  uint64_t tb, te;/* whole-sample read begin / end */
  uint32_t task_before, task_after; /* only filled by the RAW_OFF probe */
  int validity;   /* BW_SAMPLE_* */
  struct bw_core c[3]; /* core0 @0xfdab0000, core1 @0xfdac0000, core2 @0xfdad0000 */
};
/* sample validity vocabulary, reported as text in the csv */
#define BW_SAMPLE_OK 0
#define BW_SAMPLE_UNCHECKED 1   /* no task_after read was issued */
#define BW_SAMPLE_CROSSED 2     /* task counter changed across the raw read */
#define BW_SAMPLE_AMBIGUOUS 3   /* counter 0: idle and task 0 are indistinguishable */
static int g_bw_enabled, g_bw_min_eff = 4000, g_bw_nth = 1, g_bw_seen;
static int g_bw_op_idx = -1, g_bw_require_all;
static volatile int g_bw_run;
static int g_bw_n, g_bw_cap = 400000, g_bw_fd = -1;
static volatile uint32_t* g_bw_regs;
static struct bw_sample* g_bw_buf;
static pthread_t g_bw_th;
/* Lifecycle: IDLE -> STARTING -> RUNNING -> STOPPING -> IDLE.  Every transition
 * happens under g_bw_lifecycle_lock; STARTING and STOPPING both block a new run,
 * so a start can never overlap another start's setup or another run's teardown
 * (which would reuse the same thread handle, buffer and stop flag). */
enum { BW_IDLE = 0, BW_STARTING, BW_RUNNING, BW_STOPPING };
static int g_bw_state = BW_IDLE;    /* guarded by g_bw_lifecycle_lock */
static long g_bw_start_delay_ns;    /* selftest only: widen the STARTING window */
static long g_bw_stop_delay_ns;     /* selftest only: widen the STOPPING window */
/* A run belongs to ONE ioctl invocation, identified by a cookie that is unique
 * per call.  Comparing the user argument pointer was not isolation: two
 * concurrent calls can pass the same buffer address. */
static uint64_t g_bw_owner_cookie;
static uint64_t g_bw_cookie_seq;
static long g_bw_owner_conflicts;
static pthread_mutex_t g_bw_lifecycle_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_bw_all_cores;
static int g_bw_minimal;
static long g_bw_nsleep;
static int g_bw_select = 5;      /* 0=ts,1=CNA,2=CORE,3=DPU,4=DPU_RDMA,5=all */
static long g_bw_spin_ns;        /* busy-wait floor between samples (0=off) */
static long g_bw_phase_ns;       /* delay after the real kick before sampling */
static int g_bw_edge;            /* sample once per PC task-counter change */
/* P0 additions: one budget, one pacer, one lifecycle. */
static long g_bw_poll_ns = 5000; /* min interval between edge-mode poll reads */
static long g_bw_max_reads;      /* hard MMIO read budget per run (0=unlimited) */
static long g_bw_power_check_ns = 500000; /* power guard period (0=off) */
static int g_bw_raw_off = -1;    /* EPERF_RKNPU_BLOCKWATCH_RAW_OFF: single reg */
static int g_bw_raw_off2 = -1;   /* optional second register, same rules */
static int g_bw_raw_mode;        /* 0 off, 1 vetted, 2 experimental */
static int g_bw_raw_experimental;/* allow non-vetted (but not forbidden) offsets */
static int g_bw_raw_core;        /* core index for RAW_OFF */
static int g_bw_raw_rejected;    /* RAW_OFF was set but failed validation */
static int g_bw_task_check = 1;  /* RAW_OFF: read task status after the target */
static volatile int g_bw_ready;  /* sampler thread is up, waiting to be armed */
static volatile int g_bw_armed;  /* caller is about to enter the real ioctl */
static volatile uint64_t g_bw_submit_enter_ns; /* monotonic time stored by bw_arm() */
static long g_bw_reads;          /* MMIO reads issued in this run */
static long g_bw_polls;          /* edge-mode poll reads in this run */
static int g_bw_budget_hit;      /* stopped because the read budget ran out */
static const char* g_bw_stop_reason = "run_flag"; /* cap|budget|power|run_flag */
static int g_bw_power_lost;      /* power guard saw the domain drop */
static int g_bw_power_checks;
static int g_bw_power_fd = -1;
static char g_bw_power_path[256] = "/sys/kernel/debug/rknpu/power";
/* Host selftest mode: replace the /dev/mem window with a calloc'd buffer and
 * drive a synthetic task counter, so the sampler's pacing, budget, arm and
 * stop logic can be exercised without a board.  Never set on hardware. */
static int g_bw_fake;
static long g_bw_fake_busy_after_ns;
/* Latency controls.  Each is a separate, measurable variant; none of them may
 * weaken the power guard. */
static int g_bw_prefault;        /* warm the MMIO mapping + sample buffer early */
static int g_bw_preopen_power;   /* open the power fd before the run is armed */
static int g_bw_pin_cpu = -1;    /* pin the sampler thread to one CPU */
/* Latency breakdown, all CLOCK_MONOTONIC_RAW, 0 = not reached. */
static uint64_t g_bw_t_arm_observed, g_bw_t_phase_end;
static uint64_t g_bw_t_power_begin, g_bw_t_power_end;
/* real ioctl entry / return, CLOCK_MONOTONIC_RAW, same domain as submit_enter */
static uint64_t g_bw_ioctl_start_ns, g_bw_ioctl_end_ns; /* selftest only: module turns busy at T */
/* selftest only: emulate a register whose host read-back lags k writing tasks */
static long g_bw_fake_task = -1;
static long g_bw_fake_bank_lag = -1;

/* Read-only interrupt words.  INTERRUPT_STATUS (0x0028) and INTERRUPT_RAW_STATUS
 * (0x002c) are the only PC-page offsets the sampler may read.  The driver reads
 * INTERRUPT_STATUS in its own IRQ path, so reading is the normal access pattern.
 * What is NOT established on this board is that a sampler read leaves the latched
 * bits intact; that is checked empirically (dmesg must not gain "invalid irq
 * status" and no job may time out).  INTERRUPT_MASK (0x0020) and INTERRUPT_CLEAR
 * (0x0024) are deliberately not readable here. */
static int bw_raw_offset_irq_status(int off) {
  return off == 0x0028 || off == 0x002c;
}

/* RAW_OFF must name a register we are willing to touch during execution.
 *
 * Page-level rules are not enough: 0x4104 is the DPU LUT access data port and
 * 0x1ffc is not a register anybody has verified.  So there are three tiers:
 *
 *   vetted        the four module S_STATUS words this hook already reads on
 *                 hardware -- the only offsets with production evidence;
 *   experimental  any other 32-bit aligned word that is not known-dangerous;
 *                 needs EPERF_RKNPU_BLOCKWATCH_RAW_OFF_EXPERIMENTAL=1 and is
 *                 reported as such, so it can never be mistaken for a supported
 *                 sampling address;
 *   forbidden     PC page (chain state), 0x2xxx (reserved/PC2), DDMA/SDMA,
 *                 S_POINTER, OPERATION_ENABLE, interrupt registers and the DPU
 *                 LUT access block -- never, experimental or not.
 */
static int bw_raw_offset_forbidden(int off) {
  if (off < 0 || (off & 3) || off >= 0x10000) return 1;
  if (bw_raw_offset_irq_status(off)) return 0;
  const int page = off >> 12;
  const int sub = off & 0xfff;
  if (page == 0x0 || page == 0x2 || page == 0x8 || page == 0x9) return 1;
  /* Per-module control words.  These exist in every block window. */
  if (sub == 0x004 || sub == 0x008) return 1;          /* S_POINTER / OP_ENABLE */
  /* Interrupt mask/clear/status/raw and TASK_CON live in the PC window
   * (0x0020..0x003c), which the page rule above already rejects.  The same
   * offsets inside CNA/CORE/DPU are ordinary config (e.g. 0x1024 is
   * CNA_DATA_SIZE1), so they must not be caught here. */
  if (page == 0x4 && sub >= 0x100 && sub <= 0x12c) return 1;  /* DPU LUT access */
  return 0;
}

static int bw_raw_offset_vetted(int off) {
  return off == 0x1000 || off == 0x3000 || off == 0x4000 || off == 0x5000;
}

/* Strict parser.  Returns 0 = rejected, 1 = vetted (read-safe on this board;
 * what the bits MEAN is not established), 2 = experimental-only,
 * 3 = read-only interrupt word.
 * Empty / whitespace / trailing junk / forbidden offsets are all rejected. */
static int bw_parse_raw_off(const char* text, int* out, int allow_experimental) {
  if (text == NULL || text[0] == '\0') return 0;
  char* end = NULL;
  errno = 0;
  const long value = strtol(text, &end, 0);
  if (errno != 0 || end == text || (end != NULL && *end != '\0')) return 0;
  if (value < 0 || value > 0xffff) return 0;
  if (bw_raw_offset_forbidden((int)value)) return 0;
  *out = (int)value;
  if (bw_raw_offset_vetted((int)value)) return 1;
  if (bw_raw_offset_irq_status((int)value)) return 3;
  return allow_experimental ? 2 : 0;
}

/* timestamp -> offset from the submit entry, or -1 when never reached */
static long long lat_of(uint64_t stamp) {
  if (stamp == 0) return -1;
  return (long long)(stamp - __atomic_load_n(&g_bw_submit_enter_ns, __ATOMIC_ACQUIRE));
}

static const char* bw_raw_mode_name(int mode) {
  switch (mode) {
    case 1: return "vetted";
    case 2: return "experimental";
    case 3: return "irq_status";
    default: return "off";
  }
}

static const char* bw_validity_name(int validity) {
  switch (validity) {
    case BW_SAMPLE_OK: return "ok";
    case BW_SAMPLE_CROSSED: return "task_crossed";
    case BW_SAMPLE_AMBIGUOUS: return "task_ambiguous";
    case BW_SAMPLE_UNCHECKED:
    default: return "unchecked";
  }
}

static int bw_power_on_open(void) {
  char b[8] = {0};
  if (g_bw_power_fd < 0) {
    g_bw_power_fd = open(g_bw_power_path, O_RDONLY);
    if (g_bw_power_fd < 0) return -1;
  }
  const ssize_t n = pread(g_bw_power_fd, b, sizeof(b) - 1, 0);
  if (n <= 0) return -1;
  return strncmp(b, "on", 2) == 0 ? 1 : 0;
}

static int bw_power_on(void) {
  const char* path = g_bw_power_path;
  FILE* f = fopen(path, "r");
  if (f == NULL) return -1;
  char b[8] = {0};
  int ok = (fgets(b, sizeof(b), f) != NULL) && strncmp(b, "on", 2) == 0;
  fclose(f);
  return ok ? 1 : 0;
}

static void bw_wait_until(uint64_t deadline) {
  while ((uint64_t)monotonic_raw_ns() < deadline) {
    __asm__ __volatile__("" ::: "memory");
  }
}

/* Single choke point for MMIO: budget accounting lives here so that no code
 * path (poll, snapshot, future marker read) can read without being charged. */
static int bw_read_u32(const volatile uint32_t* p, uint32_t* out) {
  if (g_bw_max_reads > 0 &&
      __atomic_load_n(&g_bw_reads, __ATOMIC_RELAXED) >= g_bw_max_reads) {
    return -1;
  }
  __atomic_add_fetch(&g_bw_reads, 1, __ATOMIC_RELAXED);
  *out = *p;
  return 0;
}

static void bw_nsleep_ns(long ns) {
  struct timespec d;
  if (ns <= 0) return;
  d.tv_sec = ns / 1000000000L;
  d.tv_nsec = ns % 1000000000L;
  nanosleep(&d, NULL);
}

/* Wait until `deadline`, but abandon the wait as soon as the run is cancelled.
 * A blocking wait here would keep the sampler alive for a whole poll period
 * after bw_stop() was requested, and the MMIO read that follows it would then
 * happen after the caller believes sampling has stopped. */
static int bw_wait_cancellable(uint64_t deadline) {
  for (;;) {
    if (!__atomic_load_n(&g_bw_run, __ATOMIC_ACQUIRE)) return 0;
    const uint64_t now = (uint64_t)monotonic_raw_ns();
    if (now >= deadline) return 1;
    const uint64_t remaining = deadline - now;
    if (remaining > 50000) {
      bw_nsleep_ns((long)(remaining > 200000 ? 200000 : remaining));
    } else {
      bw_wait_until(deadline);
    }
  }
}

/* Unified pacer: applies to edge and free-running mode alike.  t0 is the time
 * the sample (or poll) actually started, never the intended schedule time.
 * Returns 0 when the run was cancelled while pacing. */
static int bw_pace(uint64_t t0) {
  if (g_bw_spin_ns > 0 && !bw_wait_cancellable(t0 + (uint64_t)g_bw_spin_ns)) return 0;
  long left = g_bw_nsleep;
  while (left > 0) {
    if (!__atomic_load_n(&g_bw_run, __ATOMIC_ACQUIRE)) return 0;
    const long chunk = left > 200000 ? 200000 : left;
    bw_nsleep_ns(chunk);
    left -= chunk;
  }
  return 1;
}

/* Fill one sample.  Returns 0 when the sample is complete and may be published,
 * -1 when the read budget ran out (the half-read sample is discarded). */
static int bw_take_sample(int index) {
  struct bw_sample* s = &g_bw_buf[index];
  s->t = monotonic_raw_ns();
  if (g_bw_fake && g_bw_raw_off >= 0 && g_bw_fake_task >= 0) {
    /* Pretend the device is executing task N and that the register at RAW_OFF
     * already holds the value task (N - lag) left behind.  This is how the
     * bank-probe chain is exercised end to end without a board. */
    ((volatile uint32_t*)g_bw_regs)[0x003c / 4] = (uint32_t)g_bw_fake_task;
    if (g_bw_fake_bank_lag >= 0) {
      long source = g_bw_fake_task - g_bw_fake_bank_lag;
      if (source < 0) source = 0;
      ((volatile uint32_t*)g_bw_regs)[g_bw_raw_off / 4] = (uint32_t)(0x1000 + source);
    }
  }
  if (g_bw_fake && g_bw_fake_busy_after_ns > 0) {
    /* Deterministic synthetic module: idle before T, busy from T onwards, so
     * the whole sweep -> CSV -> analyzer chain can be validated end to end. */
    const uint64_t elapsed =
        (uint64_t)s->t - __atomic_load_n(&g_bw_submit_enter_ns, __ATOMIC_ACQUIRE);
    ((volatile uint32_t*)g_bw_regs)[0x1000 / 4] =
        elapsed >= (uint64_t)g_bw_fake_busy_after_ns ? 1u : 0u;
  }
  s->tb = s->t;
  s->task_before = s->task_after = 0;
  s->validity = BW_SAMPLE_UNCHECKED;
  if (g_bw_raw_off >= 0) {
    /* Route A/B single-register probe: task status, the target register, and
     * (by default) task status again, so a read that straddles a task boundary
     * is detectable instead of being silently attributed to task_before. */
    const int k = g_bw_raw_core;
    const volatile uint32_t* r = g_bw_regs + (size_t)k * (0x10000 / 4);
    struct bw_core* c = &s->c[0];
    c->cna = c->core = c->dpu = c->rdma = 0;
    c->gb = monotonic_raw_ns();
    if (bw_read_u32(&r[0x003c / 4], &c->ts) != 0) return -1;
    s->task_before = c->ts & 0xfffu;
    if (bw_read_u32(&r[g_bw_raw_off / 4], &c->cna) != 0) return -1;
    if (g_bw_raw_off2 >= 0) {
      if (bw_read_u32(&r[g_bw_raw_off2 / 4], &c->core) != 0) return -1;
    }
    if (g_bw_task_check) {
      uint32_t after = 0;
      if (bw_read_u32(&r[0x003c / 4], &after) != 0) return -1;
      s->task_after = after & 0xfffu;
      if (s->task_after != s->task_before) s->validity = BW_SAMPLE_CROSSED;
      else if (s->task_before == 0) s->validity = BW_SAMPLE_AMBIGUOUS;
      else s->validity = BW_SAMPLE_OK;
    } else {
      s->task_after = 0xffffffffu;
      s->validity = BW_SAMPLE_UNCHECKED;
    }
    c->ge = monotonic_raw_ns();
    s->te = c->ge;
    return 0;
  }
  const int nc = g_bw_all_cores ? 3 : 1;
  for (int k = 0; k < nc; ++k) {
    const volatile uint32_t* r = g_bw_regs + (size_t)k * (0x10000 / 4);
    struct bw_core* c = &s->c[k];
    c->cna = c->core = c->dpu = c->rdma = 0;
    c->gb = monotonic_raw_ns();
    if (bw_read_u32(&r[0x003c / 4], &c->ts) != 0) return -1;
    if (!g_bw_minimal) {
      if (g_bw_select == 1 || g_bw_select == 5) {
        if (bw_read_u32(&r[0x1000 / 4], &c->cna) != 0) return -1;
      }
      if (g_bw_select == 2 || g_bw_select == 5) {
        if (bw_read_u32(&r[0x3000 / 4], &c->core) != 0) return -1;
      }
      if (g_bw_select == 3 || g_bw_select == 5) {
        if (bw_read_u32(&r[0x4000 / 4], &c->dpu) != 0) return -1;
      }
      if (g_bw_select == 4 || g_bw_select == 5) {
        if (bw_read_u32(&r[0x5000 / 4], &c->rdma) != 0) return -1;
      }
    }
    c->ge = monotonic_raw_ns();
  }
  s->te = monotonic_raw_ns();
  return 0;
}

/* Cheap, coarse power guard: sampled at most every power_check_ns. */
static int bw_power_guard(uint64_t* next_check) {
  if (g_bw_power_check_ns <= 0) return 0;
  const uint64_t now = monotonic_raw_ns();
  if (now < *next_check) return 0;
  *next_check = now + (uint64_t)g_bw_power_check_ns;
  g_bw_power_checks++;
  if (bw_power_on_open() != 1) {
    g_bw_power_lost = 1;
    return -1;
  }
  return 0;
}

static void* bw_thread(void* arg) {
  (void)arg;
  if (g_bw_pin_cpu >= 0) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(g_bw_pin_cpu, &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }
  __atomic_store_n(&g_bw_ready, 1, __ATOMIC_RELEASE);
  /* Wait to be armed: the phase must be measured from the real kick, so
   * sleeping here would silently push the submit behind the phase delay. */
  while (__atomic_load_n(&g_bw_armed, __ATOMIC_ACQUIRE) == 0) {
    if (!__atomic_load_n(&g_bw_run, __ATOMIC_RELAXED)) return NULL;
    __asm__ __volatile__("" ::: "memory");
  }
  const uint64_t kick = __atomic_load_n(&g_bw_submit_enter_ns, __ATOMIC_ACQUIRE);
  g_bw_t_arm_observed = (uint64_t)monotonic_raw_ns();
  if (g_bw_phase_ns > 0) bw_wait_until(kick + (uint64_t)g_bw_phase_ns);
  g_bw_t_phase_end = (uint64_t)monotonic_raw_ns();
  uint64_t next_power = 0; /* 0 => the very first iteration checks, so no MMIO
                            * read ever happens before a power probe */
  uint64_t next_poll = (uint64_t)monotonic_raw_ns();
  g_bw_t_power_begin = (uint64_t)monotonic_raw_ns();
  const int power_rc = bw_power_guard(&next_power);
  g_bw_t_power_end = (uint64_t)monotonic_raw_ns();
  if (power_rc != 0) {
    g_bw_stop_reason = "power";
    return NULL;
  }

  if (g_bw_edge) {
    uint32_t last_lo = 0xffffffffu;
    while (__atomic_load_n(&g_bw_run, __ATOMIC_RELAXED)) {
      if (g_bw_n >= g_bw_cap) { g_bw_stop_reason = "cap"; break; }
      if (bw_power_guard(&next_power) != 0) { g_bw_stop_reason = "power"; break; }
      if (g_bw_poll_ns > 0) {
        /* Poll floor: the counter read is charged to the budget and spaced, so
         * an unchanged counter cannot drive an unbounded MMIO loop.  The wait
         * must be cancellable: stopping the sampler has to stop the MMIO too. */
        if (!bw_wait_cancellable(next_poll)) break;
        next_poll = (uint64_t)monotonic_raw_ns() + (uint64_t)g_bw_poll_ns;
        /* re-check everything before the read the wait was standing in for */
        if (!__atomic_load_n(&g_bw_run, __ATOMIC_ACQUIRE)) break;
        if (bw_power_guard(&next_power) != 0) { g_bw_stop_reason = "power"; break; }
      }
      if (g_bw_fake) {
        /* advance the synthetic task counter every third poll so that both the
         * "changed" and the "unchanged -> continue" paths are exercised */
        static unsigned fake_polls;
        if ((++fake_polls % 3u) == 0) {
          ((volatile uint32_t*)g_bw_regs)[0x003c / 4] =
              (uint32_t)((fake_polls / 3u) & 0xfffu);
        }
      }
      const volatile uint32_t* r0 = g_bw_regs;
      const uint64_t poll_t0 = (uint64_t)monotonic_raw_ns();
      uint32_t v0 = 0;
      __atomic_add_fetch(&g_bw_polls, 1, __ATOMIC_RELAXED);
      if (bw_read_u32(&r0[0x003c / 4], &v0) != 0) {
        g_bw_budget_hit = 1; g_bw_stop_reason = "budget"; break;
      }
      const uint32_t lo = v0 & 0xfffu;
      if (lo == last_lo) continue;
      last_lo = lo;
      const int i = g_bw_n;
      if (bw_take_sample(i) != 0) {
        g_bw_budget_hit = 1; g_bw_stop_reason = "budget"; break;
      }
      g_bw_n = i + 1;
      if (!bw_pace(poll_t0)) break;
    }
    return NULL;
  }

  while (__atomic_load_n(&g_bw_run, __ATOMIC_RELAXED)) {
    if (g_bw_n >= g_bw_cap) { g_bw_stop_reason = "cap"; break; }
    if (bw_power_guard(&next_power) != 0) { g_bw_stop_reason = "power"; break; }
    const int i = g_bw_n;
    const uint64_t sample_t0 = monotonic_raw_ns();
    if (bw_take_sample(i) != 0) {
      g_bw_budget_hit = 1; g_bw_stop_reason = "budget"; break;
    }
    g_bw_n = i + 1;
    if (!bw_pace(sample_t0)) break;
    /* no per-sample debugfs read: the blocking submit ioctl holds the power
     * reference, and the coarse guard above re-checks it periodically. */
  }
  return NULL;
}

static int bw_state_is(int state) {
  pthread_mutex_lock(&g_bw_lifecycle_lock);
  const int match = g_bw_state == state;
  pthread_mutex_unlock(&g_bw_lifecycle_lock);
  return match;
}

static int bw_start(uint64_t cookie) {
  pthread_mutex_lock(&g_bw_lifecycle_lock);
  if (g_bw_state != BW_IDLE) {   /* STARTING/STOPPING are not slots either */
    __atomic_add_fetch(&g_bw_owner_conflicts, 1, __ATOMIC_RELAXED);
    pthread_mutex_unlock(&g_bw_lifecycle_lock);
    return -1;
  }
  g_bw_state = BW_STARTING;
  g_bw_owner_cookie = cookie;
  pthread_mutex_unlock(&g_bw_lifecycle_lock);
  if (g_bw_start_delay_ns > 0) bw_nsleep_ns(g_bw_start_delay_ns);
  void* m = NULL;
  if (g_bw_fake) {
    m = calloc(0x30000 / 4, sizeof(uint32_t));
    if (m == NULL) {
      pthread_mutex_lock(&g_bw_lifecycle_lock);
      g_bw_state = BW_IDLE;
      pthread_mutex_unlock(&g_bw_lifecycle_lock);
      return -1;
    }
    /* A running task rather than an idle core: the raw probe's task_before /
     * task_after check must be able to distinguish "task 7" from "idle". */
    ((uint32_t*)m)[0x003c / 4] = 7u;
  } else {
    if (bw_power_on() != 1) { /* never touch MMIO with the domain off */
      pthread_mutex_lock(&g_bw_lifecycle_lock);
      g_bw_state = BW_IDLE;
      pthread_mutex_unlock(&g_bw_lifecycle_lock);
      return -1;
    }
    g_bw_fd = open("/dev/mem", O_RDONLY | O_SYNC);
    if (g_bw_fd < 0) {
      pthread_mutex_lock(&g_bw_lifecycle_lock);
      g_bw_state = BW_IDLE;
      pthread_mutex_unlock(&g_bw_lifecycle_lock);
      return -1;
    }
    m = mmap(NULL, 0x30000, PROT_READ, MAP_SHARED, g_bw_fd, (off_t)0xfdab0000);
    if (m == MAP_FAILED) {
      close(g_bw_fd); g_bw_fd = -1;
      pthread_mutex_lock(&g_bw_lifecycle_lock);
      g_bw_state = BW_IDLE;
      pthread_mutex_unlock(&g_bw_lifecycle_lock);
      return -1;
    }
  }
  g_bw_regs = (volatile uint32_t*)m;   /* whole 3-core window starts here */
  g_bw_buf = (struct bw_sample*)malloc((size_t)g_bw_cap * sizeof(*g_bw_buf));
  if (g_bw_buf == NULL) {
    if (g_bw_fake) free(m); else munmap(m, 0x30000);
    close(g_bw_fd); g_bw_fd = -1;
    pthread_mutex_lock(&g_bw_lifecycle_lock);
    g_bw_state = BW_IDLE;
    pthread_mutex_unlock(&g_bw_lifecycle_lock);
    return -1;
  }
  g_bw_t_arm_observed = g_bw_t_phase_end = 0;
  g_bw_t_power_begin = g_bw_t_power_end = 0;
  g_bw_ioctl_start_ns = g_bw_ioctl_end_ns = 0;
  if (g_bw_preopen_power) (void)bw_power_on_open();
  if (g_bw_prefault && !g_bw_fake) {
    /* Warm the device mapping and the first few sample slots BEFORE the run is
     * armed, so page-table setup and first-touch faults do not land inside the
     * measured window.  The read is charged to the budget like any other. */
    uint32_t sink = 0;
    (void)bw_read_u32(&g_bw_regs[0x003c / 4], &sink);
    memset(g_bw_buf, 0, sizeof(*g_bw_buf) * (g_bw_cap < 8 ? (size_t)g_bw_cap : 8u));
  }
  g_bw_n = 0;
  g_bw_reads = 0;
  g_bw_polls = 0;
  g_bw_budget_hit = 0;
  g_bw_power_lost = 0;
  g_bw_power_checks = 0;
  g_bw_stop_reason = "run_flag";
  __atomic_store_n(&g_bw_armed, 0, __ATOMIC_RELEASE);
  __atomic_store_n(&g_bw_ready, 0, __ATOMIC_RELEASE);
  __atomic_store_n(&g_bw_run, 1, __ATOMIC_RELAXED);
  if (pthread_create(&g_bw_th, NULL, bw_thread, NULL) != 0) {
    __atomic_store_n(&g_bw_run, 0, __ATOMIC_RELAXED);
    free(g_bw_buf); munmap(m, 0x30000); close(g_bw_fd); g_bw_fd = -1;
    pthread_mutex_lock(&g_bw_lifecycle_lock);
    g_bw_state = BW_IDLE;
    pthread_mutex_unlock(&g_bw_lifecycle_lock);
    return -1;
  }
  pthread_mutex_lock(&g_bw_lifecycle_lock);
  g_bw_state = BW_RUNNING;
  pthread_mutex_unlock(&g_bw_lifecycle_lock);
  /* Handshake: thread is up and parked on the arm flag.  No sample is required
   * here, so a phase delay never runs before the submit. */
  while (__atomic_load_n(&g_bw_ready, __ATOMIC_ACQUIRE) == 0) sched_yield();
  return 0;
}

/* Called immediately before entering the real submit ioctl of the owned call.
 * The phase delay is measured from this instant, which is the last point the
 * hook can observe in user space: it is NOT the hardware kick (the syscall,
 * driver preparation and queueing all happen after it).  Reports call this
 * submit-relative time, never "kick".
 *
 * Any other ioctl that reaches this point while a run is active is a conflict:
 * it must not move the time origin of a run it does not own. */
/* Ownership probe for the ioctl epilogue.  The wrapper must know whether this
 * call owns the run BEFORE it treats the argument as an rknpu_submit: a foreign
 * ioctl can legitimately carry a 4-byte or unmapped buffer. */
static void bw_stop(uint64_t cookie, int result, long long hw_elapse, long long wall_ns);

static int bw_owns(uint64_t cookie) {
  pthread_mutex_lock(&g_bw_lifecycle_lock);
  const int owns = g_bw_state == BW_RUNNING && cookie == g_bw_owner_cookie;
  pthread_mutex_unlock(&g_bw_lifecycle_lock);
  return owns;
}

/* Single entry point for "this ioctl returned".  It checks ownership first and
 * only then reads the submit-specific fields, which is the property the foreign
 * ioctl regression test asserts. */
/* Entry and return of the real ioctl, on the same clock as submit_enter.  The
 * return is the user-visible completion, so the sampler timeline can be compared
 * with it without mixing clock domains. */
static void bw_note_ioctl(uint64_t start_ns, uint64_t end_ns) {
  __atomic_store_n(&g_bw_ioctl_start_ns, start_ns, __ATOMIC_RELEASE);
  __atomic_store_n(&g_bw_ioctl_end_ns, end_ns, __ATOMIC_RELEASE);
}

static void bw_stop_from_ioctl(uint64_t cookie, int is_submit, const void* argument,
                               int result, uint64_t start_ns, uint64_t end_ns) {
  if (!bw_owns(cookie)) return;   /* never dereference a foreign argument */
  bw_note_ioctl(start_ns, end_ns);
  const struct rknpu_submit* pba =
      (is_submit && argument != NULL) ? (const struct rknpu_submit*)argument : NULL;
  bw_stop(cookie, result, pba ? (long long)pba->hw_elapse_time : -1,
          (long long)(end_ns - start_ns));
}

static void bw_arm(uint64_t cookie) {
  pthread_mutex_lock(&g_bw_lifecycle_lock);
  const int owns = g_bw_state == BW_RUNNING && cookie == g_bw_owner_cookie;
  const int already = __atomic_load_n(&g_bw_armed, __ATOMIC_ACQUIRE) != 0;
  if (owns && !already) {
    __atomic_store_n(&g_bw_submit_enter_ns, monotonic_raw_ns(), __ATOMIC_RELEASE);
    __atomic_store_n(&g_bw_armed, 1, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&g_bw_lifecycle_lock);
  if (!owns) __atomic_add_fetch(&g_bw_owner_conflicts, 1, __ATOMIC_RELAXED);
}

/* Only the owning ioctl may end the run: the claim is taken under the lock so a
 * concurrent foreign call cannot free the buffers out from under the sampler. */
static void bw_stop(uint64_t cookie, int result, long long hw_elapse, long long wall_ns) {
  pthread_mutex_lock(&g_bw_lifecycle_lock);
  const int owns = g_bw_state == BW_RUNNING && cookie == g_bw_owner_cookie;
  if (owns) g_bw_state = BW_STOPPING;   /* claim the teardown, block new starts */
  pthread_mutex_unlock(&g_bw_lifecycle_lock);
  if (!owns) {
    __atomic_add_fetch(&g_bw_owner_conflicts, 1, __ATOMIC_RELAXED);
    return;
  }
  if (g_bw_stop_delay_ns > 0) bw_nsleep_ns(g_bw_stop_delay_ns);
  __atomic_store_n(&g_bw_run, 0, __ATOMIC_RELEASE);
  pthread_join(g_bw_th, NULL);
  const char* path = getenv("EPERF_RKNPU_BLOCKWATCH_FILE");
  if (path != NULL) {
    FILE* f = fopen(path, "w");
    if (f != NULL) {
      const int nc = (g_bw_raw_off >= 0) ? 1 : (g_bw_all_cores ? 3 : 1);
      /* t_ns / t_end_ns bracket the whole sample.  c*_gb_ns / c*_ge_ns bracket a
       * *group* of reads (1..5 registers), not a single register: per-register
       * brackets would cost two clock reads per MMIO access. */
      fprintf(f, "t_ns,t_end_ns");
      if (g_bw_raw_off >= 0) {
        fprintf(f, ",c%d_ts,c%d_raw,c%d_raw2,c%d_gb_ns,c%d_ge_ns", g_bw_raw_core,
                g_bw_raw_core, g_bw_raw_core, g_bw_raw_core, g_bw_raw_core);
      } else {
        for (int k = 0; k < nc; ++k)
          fprintf(f, ",c%d_gb_ns,c%d_ge_ns,c%d_ts,c%d_cna,c%d_core,c%d_dpu,c%d_rdma",
                  k, k, k, k, k, k, k);
      }
      fprintf(f, ",submit_enter_ns,task_before,task_after,sample_validity\n");
      for (int i = 0; i < g_bw_n; ++i) {
        const struct bw_sample* s = &g_bw_buf[i];
        fprintf(f, "%llu,%llu", (unsigned long long)s->t, (unsigned long long)s->te);
        if (g_bw_raw_off >= 0) {
          fprintf(f, ",0x%08x,0x%08x,0x%08x,%llu,%llu", s->c[0].ts, s->c[0].cna,
                  s->c[0].core, (unsigned long long)s->c[0].gb,
                  (unsigned long long)s->c[0].ge);
        } else {
          for (int k = 0; k < nc; ++k)
            fprintf(f, ",%llu,%llu,0x%08x,0x%08x,0x%08x,0x%08x,0x%08x",
                    (unsigned long long)s->c[k].gb, (unsigned long long)s->c[k].ge,
                    s->c[k].ts, s->c[k].cna, s->c[k].core, s->c[k].dpu, s->c[k].rdma);
        }
        fprintf(f, ",%llu,%u,%u,%s\n",
                (unsigned long long)__atomic_load_n(&g_bw_submit_enter_ns, __ATOMIC_ACQUIRE),
                s->task_before, s->task_after, bw_validity_name(s->validity));
      }
      fclose(f);
    }
  }
  {
    uint32_t last = g_bw_n > 0 ? g_bw_buf[g_bw_n - 1].c[0].ts : 0;
    fprintf(stderr,
            "BLOCKWATCH_META result=%d hw_elapse_ns=%lld wall_ns=%lld samples=%d cap=%d "
            "overflow=%d c0_counter0=%u c0_counter1=%u all_cores=%d probe=%d dedup=%d "
            "mmio_reads=%ld polls=%ld budget_hit=%d power_lost=%d power_checks=%d "
            "submit_enter_to_first_sample_ns=%lld stop_reason=%s "
            "lat_arm_observed_ns=%lld lat_phase_end_ns=%lld lat_power_begin_ns=%lld "
            "lat_power_end_ns=%lld lat_first_read_ns=%lld "
            "lat_ioctl_start_ns=%lld lat_ioctl_return_ns=%lld\n",
            result, hw_elapse, wall_ns, g_bw_n, g_bw_cap,
            g_bw_n >= g_bw_cap ? 1 : 0, last & 0xfffu, (last >> 16) & 0xfffu,
            g_bw_all_cores, getenv("EPERF_RKNPU_PROBE") != NULL,
            getenv("EPERF_RKNPU_DEDUP") != NULL,
            g_bw_reads, g_bw_polls, g_bw_budget_hit, g_bw_power_lost, g_bw_power_checks,
            g_bw_n > 0 ? (long long)(g_bw_buf[0].t - __atomic_load_n(&g_bw_submit_enter_ns, __ATOMIC_ACQUIRE)) : -1LL,
            g_bw_stop_reason,
            lat_of(g_bw_t_arm_observed), lat_of(g_bw_t_phase_end),
            lat_of(g_bw_t_power_begin), lat_of(g_bw_t_power_end),
            g_bw_n > 0 ? (long long)(g_bw_buf[0].c[0].gb - __atomic_load_n(&g_bw_submit_enter_ns, __ATOMIC_ACQUIRE)) : -1LL,
            lat_of(g_bw_ioctl_start_ns), lat_of(g_bw_ioctl_end_ns));
    fprintf(stderr, "BLOCKWATCH_CONFIG select=%d spin_ns=%ld phase_ns=%ld nsleep_ns=%ld minimal=%d edge=%d "
            "poll_ns=%ld max_reads=%ld power_check_ns=%ld raw_off=%s raw_core=%d "
            "raw_rejected=%d raw_mode=%s raw_off2=%s task_check=%d owner_conflicts=%ld "
            "prefault=%d preopen_power=%d pin_cpu=%d\n",
            g_bw_select, g_bw_spin_ns, g_bw_phase_ns, g_bw_nsleep, g_bw_minimal, g_bw_edge,
            g_bw_poll_ns, g_bw_max_reads, g_bw_power_check_ns,
            g_bw_raw_off >= 0 ? "on" : "off", g_bw_raw_core,
            g_bw_raw_rejected,
            bw_raw_mode_name(g_bw_raw_mode),
            g_bw_raw_off2 >= 0 ? "on" : "off",
            g_bw_task_check, g_bw_owner_conflicts,
            g_bw_prefault, g_bw_preopen_power, g_bw_pin_cpu);
  }
  fflush(stderr);
  if (g_bw_fake) free((void*)g_bw_regs); else munmap((void*)g_bw_regs, 0x30000);
  close(g_bw_fd); g_bw_fd = -1; g_bw_regs = NULL;
  if (g_bw_power_fd >= 0) { close(g_bw_power_fd); g_bw_power_fd = -1; }
  free(g_bw_buf); g_bw_buf = NULL;
  pthread_mutex_lock(&g_bw_lifecycle_lock);
  g_bw_state = BW_IDLE;                 /* only now may the next run start */
  g_bw_owner_cookie = 0;
  pthread_mutex_unlock(&g_bw_lifecycle_lock);
}

/* Duplicate-write elimination (causal ablation): replace a non-side-effect
 * NPUOP whose value equals the last value written to the same (block,reg) with
 * OP_NONE, preserving regcmd length and task count.  Side-effect registers
 * (*_S_POINTER, OP_ENABLE, LUT_ACCESS_*, PC/interrupt) are never touched. */
static int g_ded_enabled, g_ded_min_eff = 4000, g_ded_nth = 1, g_ded_seen, g_ded_done, g_ded_exit, g_ded_reported;
/* Duplicate-write detection used to be a flat "last value per (block,reg)" map.
 * That is wrong for a ping-pong register file: two writes with the same value
 * can target DIFFERENT banks, so the second one is a first initialization of the
 * other bank, not a redundant repeat.  Two changes make the predicate safe:
 *
 *   - the key includes a generation that is bumped by every bank-affecting write
 *     (S_POINTER, OPERATION_ENABLE, LUT access) and by every PC/chain write, so
 *     a match can only happen with no boundary in between;
 *   - the table is small, direct-mapped and therefore LOSES entries on
 *     collisions.  A lost entry means "not redundant", which is the safe
 *     direction: we may skip a usable slot, we never delete a needed write.
 *
 * The old `value + 1` sentinel also wrapped at 0xffffffff, turning a first write
 * of 0xffffffff into a false "redundant" match.  Validity is now a separate
 * field. */
#define DED_TABLE_SLOTS 8192u
struct ded_entry {
  uint32_t gen;
  uint32_t val;
  uint16_t block;
  uint16_t reg;
  uint8_t valid;
};
static struct ded_entry g_ded_tab[DED_TABLE_SLOTS];
static uint32_t g_ded_gen = 1;

/* Bank-affecting control words: a write to any of these may move the register
 * group, so no earlier observation may be reused across it. */
static int ded_bank_affecting(uint16_t reg) {
  const uint16_t sub = (uint16_t)(reg & 0xfffu);
  if (sub == 0x0004u || sub == 0x0008u) return 1;          /* S_POINTER / OP_ENABLE */
  if (reg == 0x4100u || reg == 0x4104u) return 1;          /* LUT access */
  return 0;
}

static void ded_reset(void) {
  memset(g_ded_tab, 0, sizeof(g_ded_tab));
  g_ded_gen = 1;
}

static void ded_bump_generation(void) {
  g_ded_gen++;
  if (g_ded_gen == 0) ded_reset();   /* wrap: drop everything rather than alias */
}

static unsigned ded_slot(uint32_t gen, unsigned block, unsigned reg) {
  const uint32_t h = gen * 2654435761u ^ (uint32_t)block * 8191u ^ (uint32_t)reg * 131u;
  return (unsigned)(h & (DED_TABLE_SLOTS - 1u));
}

static int ded_is_redundant(unsigned block, uint16_t reg, uint32_t val) {
  const struct ded_entry* e = &g_ded_tab[ded_slot(g_ded_gen, block, reg)];
  return e->valid && e->gen == g_ded_gen && e->block == (uint16_t)block &&
         e->reg == reg && e->val == val;
}

static void ded_note(unsigned block, uint16_t reg, uint32_t val) {
  struct ded_entry* e = &g_ded_tab[ded_slot(g_ded_gen, block, reg)];
  e->gen = g_ded_gen;
  e->block = (uint16_t)block;
  e->reg = reg;
  e->val = val;
  e->valid = 1;
}
static int g_ded_per_task;
static unsigned g_ded_words, g_ded_repl;

static int ded_block_idx(uint32_t op) {
  switch ((op >> 8) & 0xffu) {
    case 0x01: return 0;
    case 0x02: return 1;
    case 0x08: return 2;
    case 0x10: return 3;
    case 0x20: return 4;
    case 0x40: return 5;
    case 0x80: return 6;
    default: return 7;
  }
}
static int ded_side_effect(uint32_t op, uint16_t reg) {
  const uint8_t b = (uint8_t)((op >> 8) & 0xffu);
  if (b == 0x01) return 1; /* PC / trailer */
  if (reg == 0x0008 || (reg >= 0x0020 && reg <= 0x002c)) return 1; /* enable/int */
  if (reg == 0x1004 || reg == 0x3004 || reg == 0x4004 || reg == 0x5004 ||
      reg == 0x6004 || reg == 0x7004) return 1; /* S_POINTER */
  if (reg == 0x1008 || reg == 0x3008 || reg == 0x4008 || reg == 0x5008 ||
      reg == 0x6008 || reg == 0x7008) return 1; /* OP_ENABLE */
  if (reg == 0x4100 || reg == 0x4104) return 1; /* LUT access */
  return 0;
}
static int mem_sync_to_device(int fd, uint64_t obj_addr, uint64_t size) {
  struct rknpu_mem_sync sync;
  memset(&sync, 0, sizeof(sync));
  sync.flags = 1u; /* RKNPU_MEM_SYNC_TO_DEVICE */
  sync.obj_addr = obj_addr;
  sync.size = size;
  unsigned long req =
      _IOC(_IOC_READ | _IOC_WRITE, 'd', 0x40U + 0x05U, sizeof(sync));
  return real_ioctl ? real_ioctl(fd, req, &sync)
                    : (int)syscall(SYS_ioctl, fd, req, &sync);
}

struct ded_restore { uint64_t* p; uint64_t orig; };
static struct ded_restore* g_ded_res;
static size_t g_ded_rn, g_ded_rcap;
static uint64_t g_ded_bo[64];
static uint32_t g_ded_bo_size[64];
static size_t g_ded_bo_n;

static void ded_save(uint64_t* p, uint64_t orig) {
  if (g_ded_rn == g_ded_rcap) {
    g_ded_rcap = g_ded_rcap ? g_ded_rcap * 2 : 1024;
    g_ded_res = (struct ded_restore*)realloc(g_ded_res, g_ded_rcap * sizeof(*g_ded_res));
  }
  if (g_ded_res != NULL) { g_ded_res[g_ded_rn].p = p; g_ded_res[g_ded_rn].orig = orig; g_ded_rn++; }
}
static void ded_note_bo(uint64_t obj, uint32_t size) {
  for (size_t i = 0; i < g_ded_bo_n; ++i)
    if (g_ded_bo[i] == obj) return;
  if (g_ded_bo_n < 64) { g_ded_bo[g_ded_bo_n] = obj; g_ded_bo_size[g_ded_bo_n] = size; g_ded_bo_n++; }
}

/* Dedup within the EFFECTIVE subcore range using ABSOLUTE task indices, then
 * MEM_SYNC the touched regcmd BOs (they are cacheable).  Originals are saved
 * for restore via dedup_restore(). */
static void dedup_submit(int fd, struct rknpu_submit* s) {
  ded_reset();
  g_ded_words = 0; g_ded_repl = 0; g_ded_rn = 0; g_ded_bo_n = 0;
  struct memory_record* rec =
      find_record_by_object(s->task_obj_addr, s->iommu_domain_id);
  if (rec == NULL || rec->mapped_address == NULL) return;
  const struct rknpu_task* tb = (const struct rknpu_task*)rec->mapped_address;
  uint32_t seen_st[5], seen_num[5]; int seen_n = 0;
  for (int c = 0; c < 5; ++c) {
    const uint32_t st = s->subcore_task[c].task_start;
    const uint32_t num = s->subcore_task[c].task_number;
    if (num == 0) continue;
    int dup = 0;
    for (int j = 0; j < seen_n; ++j)
      if (seen_st[j] == st && seen_num[j] == num) dup = 1;
    if (dup) continue;
    if (seen_n < 5) { seen_st[seen_n] = st; seen_num[seen_n] = num; seen_n++; }
    for (uint32_t k = 0; k < num; ++k) {
      const struct rknpu_task* task = &tb[st + k];
      /* a new task starts a new observation window: cross-task "same value"
       * says nothing about the same bank */
      ded_bump_generation();
      if (task->regcfg_amount == 0) continue;
      if (g_ded_per_task) ded_reset();   /* legacy knob: whole-submit vs per-task */
      const void* host = NULL; size_t avail = 0; const char* via = "none";
      if (!resolve_regcmd_pointer(task->regcmd_addr, s->iommu_domain_id, &host, &avail, &via))
        continue;
      struct memory_record* b = g_last_resolved_record;
      unsigned cnt = task->regcfg_amount;
      if (avail < (size_t)cnt * 8u) cnt = (unsigned)(avail / 8u);
      uint64_t* w = (uint64_t*)host;
      int touched = 0;
      for (unsigned i = 0; i < cnt; ++i) {
        const uint64_t u = w[i];
        const uint16_t op = (uint16_t)((u >> 48) & 0xffffu);
        const uint16_t reg = (uint16_t)(u & 0xffffu);
        const uint32_t val = (uint32_t)((u >> 16) & 0xffffffffu);
        g_ded_words++;
        if (op == 0) continue;
        if (ded_side_effect(op, reg)) {
          /* a bank/start/chain write invalidates everything observed so far */
          if (ded_bank_affecting(reg) || ((op >> 8) & 0xffu) == 0x01u)
            ded_bump_generation();
          continue;
        }
        const unsigned blk = (unsigned)ded_block_idx(op);
        if (ded_is_redundant(blk, reg, val)) {
          ded_save(&w[i], u);
          w[i] = 0; /* OP_NONE */
          g_ded_repl++;
          touched = 1;
        } else {
          ded_note(blk, reg, val);
        }
      }
      if (touched && b != NULL) ded_note_bo(b->obj_addr, (uint32_t)b->size);
    }
  }
  for (size_t i = 0; i < g_ded_bo_n; ++i)
    mem_sync_to_device(fd, g_ded_bo[i], g_ded_bo_size[i]);
}

static void dedup_restore(int fd) {
  for (size_t i = 0; i < g_ded_rn; ++i) *g_ded_res[i].p = g_ded_res[i].orig;
  for (size_t i = 0; i < g_ded_bo_n; ++i)
    mem_sync_to_device(fd, g_ded_bo[i], g_ded_bo_size[i]);
  g_ded_rn = 0; g_ded_bo_n = 0;
}

/* ---------------------------------------------------------------------------
 * Route B marker injection, LENGTH PRESERVING.
 *
 * Inserting a word would force a rebuild of stream length, alignment, successor
 * pointer and terminal seal -- the change class that produced the timeouts in
 * SAME-CORE-OVERLAP.md.  Instead we REPLACE a slot that the duplicate-write
 * ablation already proved redundant: the same (block, register) was written
 * with the same value earlier in this submit, so dropping it cannot change the
 * computation.  All three arms therefore have identical stream length and
 * differ only in the content of one 8-byte word:
 *
 *   original : untouched baseline
 *   nop      : that slot becomes OP_NONE      (length control)
 *   marker   : that slot becomes a PPU write  (the marker itself)
 *
 * Everything is validated BEFORE anything is written; on any violation the
 * stream is left untouched and the reason is reported.
 * ------------------------------------------------------------------------- */
enum { MK_ARM_ORIGINAL = 0, MK_ARM_NOP = 1, MK_ARM_MARKER = 2 };

static int g_mk_enabled;
static int g_mk_arm = MK_ARM_ORIGINAL;
static unsigned g_mk_task, g_mk_slot;
static int g_mk_reg = 0x6010;          /* PPU DATA_CUBE_IN_HEIGHT */
static int g_mk_reg_experimental;      /* allow a non-PPU marker register */
static uint32_t g_mk_value;
static int g_mk_nth = 1, g_mk_seen, g_mk_done;
static int g_mk_applied, g_mk_saved, g_mk_reported;
static const char* g_mk_reason = "not_attempted";
/* The rewrite belongs to exactly one ioctl: restoring on any other call (or
 * syncing through a foreign fd) would undo the marker mid-submit. */
static uint64_t g_mk_owner_cookie;
static int g_mk_fd = -1;
static int g_mk_sync_apply_rc = -1, g_mk_sync_restore_rc = -1;
/* Expected original word, supplied by the planner: compared BEFORE the write, so
 * a mis-indexed plan cannot modify the wrong slot and be discovered afterwards. */
static uint64_t g_mk_expect_orig;
static int g_mk_expect_set;
static int g_mk_fake_sync;   /* selftest only: pretend the BO sync succeeded */

static int mk_sync(int fd, uint64_t addr, uint64_t size) {
  if (g_mk_fake_sync) return 0;
  return mem_sync_to_device(fd, addr, size);
}
static uint64_t g_mk_orig_word, g_mk_new_word;
static uint64_t g_mk_bo_addr, g_mk_bo_size;
static uint64_t* g_mk_word_ptr;

static const char* mk_arm_name(int arm) {
  switch (arm) {
    case MK_ARM_ORIGINAL: return "original";
    case MK_ARM_NOP: return "nop";
    case MK_ARM_MARKER: return "marker";
    default: return "invalid";
  }
}

static uint16_t marker_op_for_reg(int reg) {
  switch ((reg >> 12) & 0xf) {
    case 0x6: return 0x4001;   /* PPU */
    case 0x7: return 0x8001;   /* PPU_RDMA */
    default: return 0;
  }
}

/* Word layout verified against the captured stream:
 *   bits 63..48 op, bits 47..16 value, bits 15..0 register offset. */
static uint64_t marker_word(int reg, uint32_t value) {
  return ((uint64_t)marker_op_for_reg(reg) << 48) | ((uint64_t)value << 16) |
         (uint64_t)(uint16_t)reg;
}

/* A marker target must be a plain configuration word of a module the workload
 * never starts (PPU/PPU_RDMA).  Status, pointer, enable, interrupt and LUT
 * ports are refused. */
static int marker_reg_forbidden(int reg, int allow_experimental) {
  if (reg & 3) return 1;
  const int page = reg >> 12;
  const int sub = reg & 0xfff;
  /* Control words are never acceptable, on any page. */
  if (sub == 0x000 || sub == 0x004 || sub == 0x008) return 1;
  if (ded_side_effect(0x4001u, (uint16_t)reg)) return 1;
  /* The vetted tier is the never-started PPU pair.  Anything else is only
   * reachable through the experimental switch, and the caller reports which
   * tier was used so a result can never be mistaken for a supported target. */
  if (page == 0x6 || page == 0x7) return 0;
  return allow_experimental ? 0 : 1;
}

/* Pure decision shared by the rewrite path and the host self test.
 * Returns NULL when the injection may proceed, else a static reason. */
static const char* marker_check(uint64_t orig_word, int redundant, int marker_reg,
                                uint32_t marker_value, int arm, int allow_experimental) {
  if (arm != MK_ARM_NOP && arm != MK_ARM_MARKER) return "arm_invalid";
  const uint16_t op = (uint16_t)((orig_word >> 48) & 0xffffu);
  const uint16_t reg = (uint16_t)(orig_word & 0xffffu);
  if (op == 0) return "slot_is_nop";
  if (ded_side_effect(op, reg)) return "slot_word_side_effect";
  if (!redundant) return "slot_not_redundant";
  if (arm == MK_ARM_NOP) return NULL;
  if (marker_reg_forbidden(marker_reg, allow_experimental)) return "marker_reg_forbidden";
  if (marker_value == 0) return "marker_value_zero";
  return NULL;
}

/* Returns 0 when the original word is back in memory AND the device was told to
 * re-read it.  A failed sync is an experiment failure, not a warning. */
static int marker_restore(int fd) {
  if (!g_mk_saved || g_mk_word_ptr == NULL) return 0;
  *g_mk_word_ptr = g_mk_orig_word;
  g_mk_saved = 0;
  const int rc = mk_sync(fd, g_mk_bo_addr, g_mk_bo_size);
  g_mk_sync_restore_rc = rc;
  g_mk_word_ptr = NULL;
  return rc == 0 ? 0 : -1;
}

/* Walk the submit exactly like the dedup ablation does, track the same
 * redundancy predicate, and rewrite exactly one slot. */
static void marker_apply(int fd, struct rknpu_submit* s, uint64_t cookie) {
  g_mk_reason = "task_out_of_range";
  g_mk_applied = 0;
  g_mk_saved = 0;
  g_mk_owner_cookie = cookie;
  g_mk_fd = fd;
  g_mk_sync_apply_rc = -1;
  g_mk_sync_restore_rc = -1;
  /* Only the blocking path is supported: an async submit returns before the job
   * retires, so "restore after the ioctl" would race the hardware. */
  if ((s->flags & (1u << 1)) != 0) { g_mk_reason = "nonblocking_submit"; return; }
  struct memory_record* rec = find_record_by_object(s->task_obj_addr, s->iommu_domain_id);
  if (rec == NULL || rec->mapped_address == NULL) { g_mk_reason = "no_task_bo"; return; }
  const struct rknpu_task* tb = (const struct rknpu_task*)rec->mapped_address;

  /* Precondition: the marker register lives in a module the workload must not
   * touch.  Check every task's enable mask before rewriting anything. */
  {
    uint32_t seen_st[5], seen_num[5]; int seen_n = 0;
    for (int c = 0; c < 5; ++c) {
      const uint32_t st = s->subcore_task[c].task_start;
      const uint32_t num = s->subcore_task[c].task_number;
      if (num == 0) continue;
      int dup_task = 0;
      for (int j = 0; j < seen_n; ++j)
        if (seen_st[j] == st && seen_num[j] == num) dup_task = 1;
      if (dup_task) continue;
      if (seen_n < 5) { seen_st[seen_n] = st; seen_num[seen_n] = num; seen_n++; }
      for (uint32_t k = 0; k < num; ++k) {
        if ((tb[st + k].enable_mask & ((1u << 5) | (1u << 6))) != 0) {
          g_mk_reason = "ppu_enabled";
          return;
        }
      }
    }
  }
  ded_reset();
  unsigned flat = 0;
  int saw_target = 0;
  uint32_t seen_st[5], seen_num[5]; int seen_n = 0;
  for (int c = 0; c < 5; ++c) {
    const uint32_t st = s->subcore_task[c].task_start;
    const uint32_t num = s->subcore_task[c].task_number;
    if (num == 0) continue;
    int dup_task = 0;
    for (int j = 0; j < seen_n; ++j)
      if (seen_st[j] == st && seen_num[j] == num) dup_task = 1;
    if (dup_task) continue;
    if (seen_n < 5) { seen_st[seen_n] = st; seen_num[seen_n] = num; seen_n++; }
    for (uint32_t k = 0; k < num; ++k) {
      const struct rknpu_task* task = &tb[st + k];
      const unsigned this_flat = flat++;
      if (this_flat == g_mk_task) saw_target = 1;
      ded_bump_generation();   /* task boundary ends the window */
      if (task->regcfg_amount == 0) continue;
      const void* host = NULL; size_t avail = 0; const char* via = "none";
      if (!resolve_regcmd_pointer(task->regcmd_addr, s->iommu_domain_id, &host, &avail,
                                  &via))
        continue;
      struct memory_record* b = g_last_resolved_record;
      unsigned cnt = task->regcfg_amount;
      if (avail < (size_t)cnt * 8u) cnt = (unsigned)(avail / 8u);
      uint64_t* w = (uint64_t*)host;
      for (unsigned i = 0; i < cnt; ++i) {
        const uint64_t u = w[i];
        const uint16_t op = (uint16_t)((u >> 48) & 0xffffu);
        const uint16_t reg = (uint16_t)(u & 0xffffu);
        const uint32_t val = (uint32_t)((u >> 16) & 0xffffffffu);
        int redundant = 0;
        if (op != 0 && ded_side_effect(op, reg)) {
          if (ded_bank_affecting(reg) || ((op >> 8) & 0xffu) == 0x01u)
            ded_bump_generation();
        } else if (op != 0) {
          const unsigned blk = (unsigned)ded_block_idx(op);
          redundant = ded_is_redundant(blk, reg, val);
          ded_note(blk, reg, val);
        }
        if (this_flat != g_mk_task || i != g_mk_slot) continue;
        const char* reason = marker_check(u, redundant, g_mk_reg, g_mk_value, g_mk_arm,
                                          g_mk_reg_experimental);
        if (reason != NULL) { g_mk_reason = reason; return; }
        if (g_mk_expect_set && u != g_mk_expect_orig) {
          g_mk_reason = "unexpected_slot_word";
          return;
        }
        g_mk_orig_word = u;
        g_mk_new_word = (g_mk_arm == MK_ARM_NOP) ? 0ull : marker_word(g_mk_reg, g_mk_value);
        w[i] = g_mk_new_word;
        g_mk_word_ptr = &w[i];
        if (b != NULL) { g_mk_bo_addr = b->obj_addr; g_mk_bo_size = (uint32_t)b->size; }
        g_mk_saved = 1;
        g_mk_applied = 1;
        g_mk_reason = "applied";
        if (g_mk_bo_addr != 0) {
          const int rc = mk_sync(fd, g_mk_bo_addr, g_mk_bo_size);
          g_mk_sync_apply_rc = rc;
          if (rc != 0) {
            /* the device was never told: roll the word back, report failure */
            *g_mk_word_ptr = g_mk_orig_word;
            g_mk_saved = 0;
            g_mk_word_ptr = NULL;
            mk_sync(fd, g_mk_bo_addr, g_mk_bo_size);
            g_mk_applied = 0;
            g_mk_reason = "sync_failed_apply";
            return;
          }
        } else {
          g_mk_sync_apply_rc = 0;
        }
        return;
      }
    }
  }
  g_mk_reason = saw_target ? "slot_out_of_range" : "task_out_of_range";
}

int ioctl(int fd, unsigned long request, ...) {
  /* unique per invocation: the sampler's ownership token */
  const uint64_t call_cookie = __atomic_add_fetch(&g_bw_cookie_seq, 1, __ATOMIC_RELAXED);
  void* argument = NULL;
  if (_IOC_DIR(request) != _IOC_NONE || _IOC_SIZE(request) != 0U) {
    va_list arguments;
    va_start(arguments, request);
    argument = va_arg(arguments, void*);
    va_end(arguments);
  }

  if (in_hook) {
    if (real_ioctl != NULL) {
      return real_ioctl(fd, request, argument);
    }
    return (int)syscall(SYS_ioctl, fd, request, argument);
  }

  in_hook = 1;
  const int submit = is_rknpu_submit(request);
  struct rknpu_submit before;
  struct rknpu_task_capture* task_captures = NULL;
  size_t captured_task_count = 0;
  int task_capture_allocation_failed = 0;
  memset(&before, 0, sizeof(before));
  if (submit && argument != NULL && trace_fd >= 0) {
    memcpy(&before, argument, sizeof(before));
    const size_t task_capture_capacity = (size_t)before.task_number;
    if (task_capture_capacity > 0) {
      if (task_capture_capacity >
          SIZE_MAX / sizeof(struct rknpu_task_capture)) {
        task_capture_allocation_failed = 1;
      } else {
        task_captures = (struct rknpu_task_capture*)calloc(
            task_capture_capacity, sizeof(struct rknpu_task_capture));
        if (task_captures == NULL) {
          task_capture_allocation_failed = 1;
        } else {
          captured_task_count = capture_task_details(
              &before, task_captures, task_capture_capacity);
        }
      }
    }
  }
  if (submit && argument != NULL && !g_probe_done) {
    static int cfg_done = 0;
    if (!cfg_done) {
      cfg_done = 1;
      const char* e = getenv("EPERF_RKNPU_PROBE");
      g_probe_enabled = e != NULL && atoi(e) != 0;
      const char* l = getenv("EPERF_RKNPU_PROBE_L");
      if (l != NULL) g_probe_L = atoi(l);
      const char* m = getenv("EPERF_RKNPU_PROBE_MIN_EFF");
      if (m != NULL) g_probe_min_eff = atoi(m);
      const char* x = getenv("EPERF_RKNPU_PROBE_EXIT");
      g_probe_exit = x != NULL && atoi(x) != 0;
      const char* nth = getenv("EPERF_RKNPU_PROBE_NTH");
      if (nth != NULL) g_probe_nth = atoi(nth);
      const char* oi = getenv("EPERF_RKNPU_PROBE_OP_IDX");
      if (oi != NULL) g_probe_op_idx = atoi(oi);
      const char* ra = getenv("EPERF_RKNPU_PROBE_REQUIRE_ALL_TASKS_SAME_OP");
      g_probe_require_all = ra != NULL && atoi(ra) != 0;
      const char* ps = getenv("EPERF_RKNPU_PROBE_START");
      if (ps != NULL) g_probe_start = atoi(ps);
      const char* nz = getenv("EPERF_RKNPU_PROBE_NEUTRALIZE");
      if (nz != NULL) g_probe_neutralize = atoi(nz) != 0;
      const char* to = getenv("EPERF_RKNPU_PROBE_TIMEOUT_MS");
      if (to != NULL) g_probe_timeout_ms = atoi(to);
      g_probe_reload_l = getenv("EPERF_RKNPU_PROBE_RELOAD_L") != NULL;
    }
    if (g_probe_enabled && g_probe_L > 0) {
      if (g_probe_reload_l) {
        const char* l = getenv("EPERF_RKNPU_PROBE_L");
        if (l != NULL) g_probe_L = atoi(l);
      }
      struct rknpu_submit* s = (struct rknpu_submit*)argument;
      unsigned int scsum = 0;
      for (int i = 0; i < 5; ++i) scsum += s->subcore_task[i].task_number;
      if ((int)scsum >= g_probe_min_eff && s->task_number > 0) {
        int match = 1;
        struct memory_record* trec = NULL;
        if (g_probe_op_idx >= 0) {
          /* verify the effective range really is the target op */
          trec = find_record_by_object(s->task_obj_addr, s->iommu_domain_id);
          if (trec == NULL || trec->mapped_address == NULL) {
            match = 0;
          } else {
            const struct rknpu_task* tb =
                (const struct rknpu_task*)trec->mapped_address;
            unsigned int st = 0, num = 0;
            for (int i = 0; i < 5; ++i)
              if (s->subcore_task[i].task_number > 0) {
                st = s->subcore_task[i].task_start;
                num = s->subcore_task[i].task_number;
                break;
              }
            unsigned int same = 0, diff = 0, k;
            for (k = 0; k < num; ++k) {
              if (tb[st + k].op_idx == (uint32_t)g_probe_op_idx) same++;
              else diff++;
            }
            if (g_probe_require_all && diff != 0) match = 0;
            if (same == 0) match = 0;
          }
        }
        g_probe_seen += match;
        if (match && g_probe_seen == g_probe_nth) {
          g_probe_orig_tn = (int)s->task_number;
          for (int i = 0; i < 5; ++i) g_probe_saved_sc[i] = s->subcore_task[i];
          unsigned int newsum = 0;
          for (int i = 0; i < 5; ++i) {
            if (s->subcore_task[i].task_number > 0) {
              unsigned int lim = s->subcore_task[i].task_number;
              unsigned int orig_start = s->subcore_task[i].task_start;
              unsigned int L = (g_probe_L > 0 && (unsigned)g_probe_L <= lim)
                                   ? (unsigned)g_probe_L : lim;
              s->subcore_task[i].task_start += (uint32_t)g_probe_start;
              s->subcore_task[i].task_number = L;
              newsum += L;
              /* EPERF_RKNPU_PROBE_NEUTRALIZE: null the removed tail so the NPU
               * cannot over-run the truncated count into still-valid tasks. */
              if (g_probe_neutralize && g_probe_start == 0 && L < lim &&
                  trec != NULL && trec->mapped_address != NULL) {
                struct rknpu_task* tbm =
                    (struct rknpu_task*)trec->mapped_address;
                for (unsigned int k = L; k < lim; ++k) {
                  tbm[orig_start + k].flags = 0;
                  tbm[orig_start + k].enable_mask = 0;
                  tbm[orig_start + k].int_mask = 0;
                  tbm[orig_start + k].int_clear = 0;
                  tbm[orig_start + k].regcfg_amount = 0;
                  tbm[orig_start + k].regcmd_addr = 0;
                }
              }
            }
          }
          s->task_number = newsum > 0 ? newsum : (uint32_t)g_probe_L;
          /* fail-stop: cap the per-submit timeout so a spill-over hang aborts in
           * seconds instead of the runtime's ~134 s, avoiding soft-reset cascades. */
          if (g_probe_timeout_ms > 0 &&
              (int)s->timeout > g_probe_timeout_ms) {
            s->timeout = (uint32_t)g_probe_timeout_ms;
          }
          g_probe_done = 1;
          g_probe_arg = argument;      /* only this ioctl's return is reported */
          g_probe_newsum = s->task_number;
          fprintf(stderr,
                  "PROBE truncate: orig_task_number=%d scsum=%u -> new_task_number=%u "
                  "timeout_ms=%u neutralize=%d\n",
                  g_probe_orig_tn, scsum, s->task_number, s->timeout,
                  g_probe_neutralize);
          fflush(stderr);
        }
      }
    }
  }
  if (submit && argument != NULL && bw_state_is(BW_IDLE)) {
    static int bwcfg = 0;
    if (!bwcfg) {
      bwcfg = 1;
      const char* e = getenv("EPERF_RKNPU_BLOCKWATCH");
      g_bw_enabled = e != NULL && atoi(e) != 0;
      const char* m = getenv("EPERF_RKNPU_BLOCKWATCH_MIN_EFF");
      if (m != NULL) g_bw_min_eff = atoi(m);
      const char* n = getenv("EPERF_RKNPU_BLOCKWATCH_NTH");
      if (n != NULL) g_bw_nth = atoi(n);
      const char* c = getenv("EPERF_RKNPU_BLOCKWATCH_CAP");
      if (c != NULL) g_bw_cap = atoi(c);
      const char* ac = getenv("EPERF_RKNPU_BLOCKWATCH_ALL_CORES");
      g_bw_all_cores = ac != NULL && atoi(ac) != 0;
      g_bw_minimal = getenv("EPERF_RKNPU_BLOCKWATCH_MINIMAL") != NULL;
      const char* ns = getenv("EPERF_RKNPU_BLOCKWATCH_NSLEEP");
      if (ns != NULL) g_bw_nsleep = atol(ns);
      const char* sel = getenv("EPERF_RKNPU_BLOCKWATCH_SELECT");
      if (sel != NULL) g_bw_select = atoi(sel);
      const char* sp = getenv("EPERF_RKNPU_BLOCKWATCH_SPIN_NS");
      if (sp != NULL) g_bw_spin_ns = atol(sp);
      const char* ph = getenv("EPERF_RKNPU_BLOCKWATCH_PHASE_NS");
      if (ph != NULL) g_bw_phase_ns = atol(ph);
      g_bw_edge = getenv("EPERF_RKNPU_BLOCKWATCH_EDGE") != NULL;
      const char* pn = getenv("EPERF_RKNPU_BLOCKWATCH_POLL_NS");
      if (pn != NULL) g_bw_poll_ns = atol(pn);
      const char* mr = getenv("EPERF_RKNPU_BLOCKWATCH_MAX_READS");
      if (mr != NULL) g_bw_max_reads = atol(mr);
      const char* pc = getenv("EPERF_RKNPU_BLOCKWATCH_POWER_CHECK_NS");
      if (pc != NULL) g_bw_power_check_ns = atol(pc);
      const char* rx = getenv("EPERF_RKNPU_BLOCKWATCH_RAW_OFF_EXPERIMENTAL");
      g_bw_raw_experimental = rx != NULL && atoi(rx) != 0;
      const char* ro = getenv("EPERF_RKNPU_BLOCKWATCH_RAW_OFF");
      if (ro != NULL) {
        int parsed = -1;
        const int tier = bw_parse_raw_off(ro, &parsed, g_bw_raw_experimental);
        if (tier != 0) {
          g_bw_raw_off = parsed;
          g_bw_raw_mode = tier;
          if (tier == 2) {
            fprintf(stderr, "BLOCKWATCH_RAW_EXPERIMENTAL offset=0x%04x "
                    "(not a vetted register; results are probe-only)\n", parsed);
            fflush(stderr);
          }
        } else {
          /* An empty or malformed value used to collapse to offset 0, i.e. the
           * PC_VERSION register.  Refuse it: raw mode stays off and the run is
           * flagged so nobody silently samples the wrong register. */
          g_bw_raw_rejected = 1;
          g_bw_raw_off = -1;
          fprintf(stderr, "BLOCKWATCH_RAW_REJECTED value=\"%s\" (allowed: "
                  "0x1xxx CNA, 0x3xxx CORE, 0x4xxx DPU, 0x5xxx DPU_RDMA, "
                  "0x6xxx PPU, 0x7xxx PPU_RDMA; 4-byte aligned)\n", ro);
          fflush(stderr);
        }
      }
      const char* ro2 = getenv("EPERF_RKNPU_BLOCKWATCH_RAW_OFF2");
      if (ro2 != NULL && ro2[0] != '\0') {
        int parsed2 = -1;
        const int tier2 = bw_parse_raw_off(ro2, &parsed2, g_bw_raw_experimental);
        if (tier2 != 0) {
          g_bw_raw_off2 = parsed2;
        } else {
          fprintf(stderr, "BLOCKWATCH_RAW_REJECTED off2 value=\"%s\"\n", ro2);
          fflush(stderr);
        }
      }
      const char* rc = getenv("EPERF_RKNPU_BLOCKWATCH_RAW_CORE");
      if (rc != NULL) g_bw_raw_core = atoi(rc);
      if (g_bw_raw_core < 0 || g_bw_raw_core > 2) g_bw_raw_core = 0;
      g_bw_prefault = getenv("EPERF_RKNPU_BLOCKWATCH_PREFAULT") != NULL;
      g_bw_preopen_power = getenv("EPERF_RKNPU_BLOCKWATCH_PREOPEN_POWER") != NULL;
      const char* pcpu = getenv("EPERF_RKNPU_BLOCKWATCH_PIN_CPU");
      if (pcpu != NULL) g_bw_pin_cpu = atoi(pcpu);
      const char* tc = getenv("EPERF_RKNPU_BLOCKWATCH_TASK_CHECK");
      if (tc != NULL) g_bw_task_check = atoi(tc) != 0;
      const char* pf = getenv("EPERF_RKNPU_BLOCKWATCH_POWER_FILE");
      if (pf != NULL && pf[0] != '\0') {
        snprintf(g_bw_power_path, sizeof(g_bw_power_path), "%s", pf);
      }
      const char* oi = getenv("EPERF_RKNPU_BLOCKWATCH_OP_IDX");
      if (oi != NULL) g_bw_op_idx = atoi(oi);
      const char* ra = getenv("EPERF_RKNPU_BLOCKWATCH_REQUIRE_ALL_TASKS_SAME_OP");
      g_bw_require_all = ra != NULL && atoi(ra) != 0;
    }
    if (g_bw_enabled) {
      struct rknpu_submit* s = (struct rknpu_submit*)argument;
      unsigned int scsum = 0;
      for (int i = 0; i < 5; ++i) scsum += s->subcore_task[i].task_number;
      if (scsum == 0) scsum = s->task_number;
      int match = 1;
      if (g_bw_op_idx >= 0) {
        struct memory_record* trec =
            find_record_by_object(s->task_obj_addr, s->iommu_domain_id);
        if (trec == NULL || trec->mapped_address == NULL) {
          match = 0;
        } else {
          const struct rknpu_task* tb =
              (const struct rknpu_task*)trec->mapped_address;
          unsigned int st = 0, num = 0, same = 0, diff = 0, k;
          for (int i = 0; i < 5; ++i)
            if (s->subcore_task[i].task_number > 0) {
              st = s->subcore_task[i].task_start;
              num = s->subcore_task[i].task_number;
              break;
            }
          for (k = 0; k < num; ++k) {
            if (tb[st + k].op_idx == (uint32_t)g_bw_op_idx) same++;
            else diff++;
          }
          if (g_bw_require_all && diff != 0) match = 0;
          if (same == 0) match = 0;
        }
      }
      if (match && (int)scsum >= g_bw_min_eff) {
        g_bw_seen++;
        if (g_bw_seen == g_bw_nth && bw_start(call_cookie) == 0) {
          fprintf(stderr, "BLOCKWATCH start: task_number=%u scsum=%u\n",
                  s->task_number, scsum);
          fflush(stderr);
        }
      }
    }
  }
  if (submit && argument != NULL && !g_mk_done) {
    static int mkcfg = 0;
    if (!mkcfg) {
      mkcfg = 1;
      const char* arm = getenv("EPERF_RKNPU_MARKER_ARM");
      const char* tk = getenv("EPERF_RKNPU_MARKER_TASK");
      const char* sl = getenv("EPERF_RKNPU_MARKER_SLOT");
      if (arm != NULL && tk != NULL && sl != NULL) {
        g_mk_enabled = 1;
        if (strcmp(arm, "nop") == 0) g_mk_arm = MK_ARM_NOP;
        else if (strcmp(arm, "marker") == 0) g_mk_arm = MK_ARM_MARKER;
        else g_mk_arm = MK_ARM_ORIGINAL;
        g_mk_task = (unsigned)strtoul(tk, NULL, 0);
        g_mk_slot = (unsigned)strtoul(sl, NULL, 0);
        const char* r = getenv("EPERF_RKNPU_MARKER_REG");
        if (r != NULL) g_mk_reg = (int)strtol(r, NULL, 0);
        const char* v = getenv("EPERF_RKNPU_MARKER_VALUE");
        if (v != NULL) g_mk_value = (uint32_t)strtoul(v, NULL, 0);
        const char* n = getenv("EPERF_RKNPU_MARKER_NTH");
        if (n != NULL) g_mk_nth = atoi(n);
        const char* rx = getenv("EPERF_RKNPU_MARKER_REG_EXPERIMENTAL");
        g_mk_reg_experimental = rx != NULL && atoi(rx) != 0;
        if (g_mk_reg_experimental) {
          fprintf(stderr, "MARKER_REG_EXPERIMENTAL: the marker register is outside the "
                  "vetted PPU pair; the run is a probe, not a supported target\n");
          fflush(stderr);
        }
        const char* eo = getenv("EPERF_RKNPU_MARKER_EXPECT_ORIG");
        if (eo != NULL && eo[0] != '\0') {
          g_mk_expect_orig = strtoull(eo, NULL, 0);
          g_mk_expect_set = 1;
        }
      }
    }
    if (g_mk_enabled) {
      g_mk_seen++;
      if (g_mk_seen == g_mk_nth) {
        g_mk_done = 1;
        if (g_mk_arm == MK_ARM_ORIGINAL) {
          g_mk_reason = "baseline";
        } else {
          marker_apply(fd, (struct rknpu_submit*)argument, call_cookie);
        }
        fprintf(stderr,
                "MARKER arm=%s task=%u slot=%u reg=0x%04x value=0x%08x applied=%d "
                "reason=%s orig=0x%016llx marker=0x%016llx sync_apply=%d expect=%s "
                "reg_tier=%s\n",
                mk_arm_name(g_mk_arm), g_mk_task, g_mk_slot, g_mk_reg, g_mk_value,
                g_mk_applied, g_mk_reason,
                (unsigned long long)g_mk_orig_word, (unsigned long long)g_mk_new_word,
                g_mk_sync_apply_rc, g_mk_expect_set ? "on" : "off",
                g_mk_reg_experimental ? "experimental" : "vetted");
        fflush(stderr);
      }
    }
  }
  if (submit && argument != NULL && !g_ded_done) {
    static int dedcfg = 0;
    if (!dedcfg) {
      dedcfg = 1;
      const char* e = getenv("EPERF_RKNPU_DEDUP");
      g_ded_enabled = e != NULL && atoi(e) != 0;
      const char* m = getenv("EPERF_RKNPU_DEDUP_MIN_EFF");
      if (m != NULL) g_ded_min_eff = atoi(m);
      const char* n = getenv("EPERF_RKNPU_DEDUP_NTH");
      if (n != NULL) g_ded_nth = atoi(n);
      const char* x = getenv("EPERF_RKNPU_DEDUP_EXIT");
      g_ded_exit = x != NULL && atoi(x) != 0;
      const char* pt = getenv("EPERF_RKNPU_DEDUP_PER_TASK");
      g_ded_per_task = pt != NULL && atoi(pt) != 0;
    }
    if (g_ded_enabled) {
      struct rknpu_submit* s = (struct rknpu_submit*)argument;
      unsigned int scsum = 0;
      for (int i = 0; i < 5; ++i) scsum += s->subcore_task[i].task_number;
      if (scsum == 0) scsum = s->task_number;
      if ((int)scsum >= g_ded_min_eff) {
        g_ded_seen++;
        if (g_ded_seen == g_ded_nth) {
          dedup_submit(fd, s);
          g_ded_done = 1;
          fprintf(stderr, "DEDUP submit: words=%u replaced=%u\n", g_ded_words, g_ded_repl);
          fflush(stderr);
        }
      }
    }
  }
  bw_arm(call_cookie); /* only the owning submit may set the time origin */
  const long long start_ns = monotonic_raw_ns();
  int result = real_ioctl != NULL ? real_ioctl(fd, request, argument)
                                  : (int)syscall(SYS_ioctl, fd, request, argument);
  const int saved_errno = errno;
  const long long end_ns = monotonic_raw_ns();
  const int call_errno = result == -1 ? saved_errno : 0;
  if (g_probe_done && argument != NULL && argument == g_probe_arg) {
    struct rknpu_submit* pa = (struct rknpu_submit*)argument;
    fprintf(stderr,
            "PROBE_RESULT L=%d orig_task_number=%d result=%d task_number=%u hw_elapse_time_ns=%lld\n",
            g_probe_L, g_probe_orig_tn, result, pa->task_number,
            (long long)pa->hw_elapse_time);
    fflush(stderr);
    dump_memory_ranges(pa, result, 1);
    if (g_probe_exit && result == 0) _exit(0);
    if (!g_probe_exit) { /* restore original submit params for the runtime */
      pa->task_number = (uint32_t)g_probe_orig_tn;
      for (int i = 0; i < 5; ++i) pa->subcore_task[i] = g_probe_saved_sc[i];
    }
    if (g_probe_reload_l) { /* allow the next inference to be probed too */
      g_probe_done = 0;
      g_probe_seen = 0;
      g_probe_arg = NULL;
    }
  }
  if (submit && result == 0 && argument != NULL &&
      !(g_probe_done && argument == g_probe_arg)) {
    /* only a submit ioctl has an rknpu_submit behind the pointer */
    dump_memory_ranges((const struct rknpu_submit*)argument, result, 0);
  }
  bw_stop_from_ioctl(call_cookie, submit, argument, result,
                     (uint64_t)start_ns, (uint64_t)end_ns);
  if (g_mk_done && g_mk_saved && !g_mk_reported && call_cookie == g_mk_owner_cookie) {
    /* only the owning ioctl may undo its own marker, and it must sync with the
     * same fd the submit used */
    g_mk_reported = 1;
    const int rc = marker_restore(g_mk_fd >= 0 ? g_mk_fd : fd);
    fprintf(stderr, "MARKER_RESTORE ok=%d sync_rc=%d applied=%d reason=%s\n",
            rc == 0 ? 1 : 0, g_mk_sync_restore_rc, g_mk_applied, g_mk_reason);
    if (rc != 0) {
      fprintf(stderr, "MARKER_SYNC_FAIL phase=restore rc=%d\n", g_mk_sync_restore_rc);
    }
    fflush(stderr);
  } else if (g_mk_done && g_mk_saved && call_cookie != g_mk_owner_cookie) {
    __atomic_add_fetch(&g_bw_owner_conflicts, 1, __ATOMIC_RELAXED);
  }
  if (g_ded_done && !g_ded_reported && g_ded_words > 0 && argument != NULL) {
    g_ded_reported = 1;
    const struct rknpu_submit* pa = (const struct rknpu_submit*)argument;
    fprintf(stderr,
            "DEDUP_RESULT words=%u replaced=%u result=%d hw_elapse_time_ns=%lld\n",
            g_ded_words, g_ded_repl, result, (long long)pa->hw_elapse_time);
    fflush(stderr);
    if (g_ded_exit && result == 0) _exit(0);
    dedup_restore(fd);
  }

  if (result == 0 && argument != NULL) {
    if (request_matches(request, 0x02U, sizeof(struct rknpu_mem_create))) {
      track_mem_create(fd, (const struct rknpu_mem_create*)argument);
      write_mem_create_record(fd, (const struct rknpu_mem_create*)argument);
    } else if (request_matches(request, 0x03U,
                               sizeof(struct rknpu_mem_map))) {
      track_mem_map(fd, (const struct rknpu_mem_map*)argument);
      write_mem_map_record(fd, (const struct rknpu_mem_map*)argument);
    } else if (request_matches(request, 0x04U,
                               sizeof(struct rknpu_mem_destroy))) {
      track_mem_destroy(fd, (const struct rknpu_mem_destroy*)argument);
    }
  }

  if (submit && argument != NULL && trace_fd >= 0) {
    struct rknpu_submit after;
    memcpy(&after, argument, sizeof(after));
    const unsigned long long sequence =
        __atomic_fetch_add(&sequence_number, 1ULL, __ATOMIC_RELAXED) + 1ULL;
    const int stage = __atomic_load_n(&current_stage, __ATOMIC_ACQUIRE);
    // Keep submit metadata bounded and independent of task_number.  Per-task
    // op_idx values are emitted by the canonical `task` records below and are
    // joined by sequence.  The previous inline task_op_idx array could exceed
    // this buffer and silently drop the entire submit record.
    unsigned int rw_total = 0;
    unsigned int rw_dt_rd = 0;
    unsigned int rw_dt_wr = 0;
    unsigned int rw_wt_rd = 0;
    unsigned int subcore_task_sum = 0;
    for (int i = 0; i < 5; ++i)
      subcore_task_sum += before.subcore_task[i].task_number;
    if (rw_counter_enabled()) {
      rw_total = query_rw_amount(fd, 17U);
      rw_dt_rd = query_rw_amount(fd, 15U);
      rw_dt_wr = query_rw_amount(fd, 14U);
      rw_wt_rd = query_rw_amount(fd, 16U);
    }
    char record[2048];
    int length = snprintf(
        record, sizeof(record),
        "{\"schema_version\":1,\"event\":\"submit\",\"sequence\":%llu,"
        "\"stage\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"fd\":%d,"
        "\"request\":\"0x%lx\",\"request_type\":%u,\"request_nr\":%u,"
        "\"request_size\":%u,\"result\":%d,\"errno\":%d,"
        "\"wall_start_ns\":%lld,\"wall_end_ns\":%lld,"
        "\"wall_elapsed_ns\":%lld,\"flags\":%u,\"timeout_ms\":%u,"
        "\"task_start\":%u,\"task_number\":%u,\"task_counter_before\":%u,"
        "\"task_counter_after\":%u,\"priority\":%d,"
        "\"task_obj_addr\":\"0x%llx\",\"task_base_addr\":\"0x%llx\","
        "\"iommu_domain_id\":%u,\"core_mask\":%u,\"core_mask_after\":%u,"
        "\"task_number_after\":%u,\"subcore_task_sum\":%u,\"fence_fd\":%d,"
        "\"hw_elapse_time_ns\":%lld,"
        "\"subcore_tasks\":[[%u,%u],[%u,%u],[%u,%u],[%u,%u],[%u,%u]],"
        "\"task_mapping_found\":%s,\"captured_task_count\":%zu,"
        "\"task_capture_complete\":%s,"
        "\"task_capture_allocation_failed\":%s,"
        "\"rw_total\":%u,\"rw_dt_rd\":%u,\"rw_dt_wr\":%u,\"rw_wt_rd\":%u,"
        "\"task_op_idx_storage\":\"task_events\"}\n",
        sequence, stage_name(stage), (long)syscall(SYS_getpid),
        (long)syscall(SYS_gettid), fd, request,
        (unsigned int)_IOC_TYPE(request), (unsigned int)_IOC_NR(request),
        (unsigned int)_IOC_SIZE(request), result, call_errno, start_ns,
        end_ns, end_ns - start_ns, before.flags, before.timeout,
        before.task_start, before.task_number, before.task_counter,
        after.task_counter, before.priority,
        (unsigned long long)before.task_obj_addr,
        (unsigned long long)before.task_base_addr, before.iommu_domain_id,
        before.core_mask, after.core_mask, after.task_number, subcore_task_sum,
        before.fence_fd, (long long)after.hw_elapse_time,
        before.subcore_task[0].task_start,
        before.subcore_task[0].task_number,
        before.subcore_task[1].task_start,
        before.subcore_task[1].task_number,
        before.subcore_task[2].task_start,
        before.subcore_task[2].task_number,
        before.subcore_task[3].task_start,
        before.subcore_task[3].task_number,
        before.subcore_task[4].task_start,
        before.subcore_task[4].task_number,
        captured_task_count > 0 ? "true" : "false", captured_task_count,
        captured_task_count == (size_t)before.task_number ? "true" : "false",
        task_capture_allocation_failed ? "true" : "false",
        rw_total, rw_dt_rd, rw_dt_wr, rw_wt_rd);
    if (length > 0 && (size_t)length < sizeof(record)) {
      write_record(record, (size_t)length);
    }
    for (size_t index = 0; index < captured_task_count; ++index) {
      const struct rknpu_task_capture* capture = &task_captures[index];
      const void* host = NULL;
      size_t available = 0;
      const char* via = "none";
      const int resolved =
          resolve_regcmd_pointer(capture->regcmd_addr, before.iommu_domain_id,
                                &host, &available, &via);
      write_task_record(sequence, index, capture, before.task_base_addr,
                        before.iommu_domain_id, resolved, via);
      if (resolved && host != NULL) {
        write_regcmd_record(sequence, index, capture, host, available);
      }
    }
  }

  free(task_captures);
  errno = saved_errno;
  in_hook = 0;
  return result;
}

void* mmap(void* address, size_t length, int protection, int flags, int fd,
           off_t offset) {
  if (in_hook) {
    return real_mmap != NULL
               ? real_mmap(address, length, protection, flags, fd, offset)
               : (void*)syscall(SYS_mmap, address, length, protection, flags,
                                fd, offset);
  }
  void* result = real_mmap != NULL
                     ? real_mmap(address, length, protection, flags, fd, offset)
                     : (void*)syscall(SYS_mmap, address, length, protection,
                                      flags, fd, offset);
  uint64_t matched_object =
      track_mmap_result(fd, (uint64_t)offset, result, length);
  write_mmap_record(fd, (uint64_t)offset, result, length, matched_object);
  return result;
}

void* mmap64(void* address, size_t length, int protection, int flags, int fd,
             off64_t offset) {
  if (in_hook) {
    return real_mmap64 != NULL
               ? real_mmap64(address, length, protection, flags, fd, offset)
               : (void*)syscall(SYS_mmap, address, length, protection, flags,
                                fd, offset);
  }
  void* result =
      real_mmap64 != NULL
          ? real_mmap64(address, length, protection, flags, fd, offset)
          : (void*)syscall(SYS_mmap, address, length, protection, flags, fd,
                           offset);
  uint64_t matched_object =
      track_mmap_result(fd, (uint64_t)offset, result, length);
  write_mmap_record(fd, (uint64_t)offset, result, length, matched_object);
  return result;
}

#ifdef EPERF_BW_SELFTEST
/* Host-only self test for the blockwatcher sampler (no board, no /dev/mem).
 * Build:  gcc -O2 -DEPERF_BW_SELFTEST -o bw_selftest rknpu_ioctl_hook.c -ldl -lpthread
 * Run:    EPERF_RKNPU_BLOCKWATCH_FILE=/tmp/bw.csv ./bw_selftest --phase-ns=20000000 ...
 * The assertions live in tests/test_blockwatch_sampler.py. */
#include <stdlib.h>
static int arg_long_opt(int argc, char** argv, const char* key, long* out) {
  const size_t n = strlen(key);
  for (int i = 1; i < argc; ++i) {
    /* base 0: "--raw-off=0x1010" must not be parsed as decimal 0, which is
     * exactly how an offset silently becomes the PC_VERSION register */
    if (strncmp(argv[i], key, n) == 0 && argv[i][n] == '=') {
      *out = strtol(argv[i] + n + 1, NULL, 0);
      return 1;
    }
  }
  return 0;
}
static long arg_long(int argc, char** argv, const char* key, long dflt) {
  long value = dflt;
  arg_long_opt(argc, argv, key, &value);
  return value;
}
static const char* arg_str(int argc, char** argv, const char* key) {
  const size_t n = strlen(key);
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], key, n) == 0 && argv[i][n] == '=') return argv[i] + n + 1;
  }
  return NULL;
}
struct bw_race_arg { uint64_t cookie; int rc; };
static void* bw_race_start(void* p) {
  struct bw_race_arg* a = (struct bw_race_arg*)p;
  a->rc = bw_start(a->cookie);
  return NULL;
}
static void* bw_race_stop(void* p) {
  struct bw_race_arg* a = (struct bw_race_arg*)p;
  bw_stop(a->cookie, 0, -1, 0);
  return NULL;
}

/* Concurrency + memory-safety probes that the sequential tests cannot reach:
 *   1. start while another start is inside its STARTING window
 *   2. start while the previous run is inside its STOPPING window
 *   3. a foreign ioctl whose argument sits on a guard page (the old code cast
 *      every ioctl argument to rknpu_submit before checking ownership)
 */
static int bw_selftest_races(void) {
  g_bw_fake = 1;              /* same fake window the normal selftest uses */
  g_bw_power_check_ns = 0;    /* no debugfs power node on a host */
  pthread_t thread;
  int failures = 0;

  /* 1. STARTING window */
  g_bw_start_delay_ns = 50000000;
  struct bw_race_arg first = {0x1111ull, 12345};
  if (pthread_create(&thread, NULL, bw_race_start, &first) != 0) return 2;
  bw_nsleep_ns(10000000);
  struct bw_race_arg second = {0x2222ull, 12345};
  const int during_starting = bw_start(second.cookie);
  pthread_join(thread, NULL);
  g_bw_start_delay_ns = 0;
  printf("SELFTEST_RACE start_during_starting=%d first_start=%d\n",
         during_starting, first.rc);
  if (during_starting != -1) failures++;
  if (first.rc != 0) failures++;
  bw_arm(first.cookie);
  bw_stop(first.cookie, 0, -1, 0);

  /* 2. STOPPING window */
  g_bw_stop_delay_ns = 50000000;
  if (bw_start(0x3333ull) != 0) return 2;
  bw_arm(0x3333ull);
  struct bw_race_arg stopper = {0x3333ull, 0};
  if (pthread_create(&thread, NULL, bw_race_stop, &stopper) != 0) return 2;
  bw_nsleep_ns(10000000);
  const int during_stopping = bw_start(0x4444ull);
  pthread_join(thread, NULL);
  g_bw_stop_delay_ns = 0;
  const int after_stopping = bw_start(0x4444ull);
  printf("SELFTEST_RACE start_during_stopping=%d start_after_stop=%d\n",
         during_stopping, after_stopping);
  if (during_stopping != -1) failures++;
  if (after_stopping != 0) failures++;
  bw_arm(0x4444ull);
  bw_stop(0x4444ull, 0, -1, 0);

  /* 3. foreign ioctl on a guard page */
  const long page = sysconf(_SC_PAGESIZE);
  unsigned char* mem = (unsigned char*)mmap(NULL, (size_t)page * 2,
                                            PROT_READ | PROT_WRITE,
                                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) return 2;
  if (mprotect(mem + page, (size_t)page, PROT_NONE) != 0) return 2;
  const void* guard = (const void*)(mem + page);
  bw_stop_from_ioctl(0xDEADull, 0, guard, 0, 0, 0);
  printf("SELFTEST_RACE foreign_ioctl_guard_page=survived\n");
  munmap(mem, (size_t)page * 2);

  printf("SELFTEST_RACE failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}

/* Regression for the foreign-ioctl dereference, driven through the REAL ioctl
 * wrapper instead of calling the helper directly:
 *
 *   - the sampler is RUNNING and armed,
 *   - a foreign request (not an rknpu submit) is issued with an argument that
 *     sits on a PROT_NONE guard page,
 *   - the wrapper must return an error without touching that pointer, must not
 *     stop or steal the run, and the owner must still be able to stop cleanly.
 */
static int bw_selftest_wrapper_probe(void) {
  g_bw_fake = 1;
  g_bw_power_check_ns = 0;
  g_bw_phase_ns = 0;
  g_bw_cap = 100000;
  int failures = 0;
  const uint64_t owner = 0xC0FFEEull;

  if (bw_start(owner) != 0) {
    printf("SELFTEST_WRAPPER start=fail\n");
    return 2;
  }
  bw_arm(owner);
  bw_nsleep_ns(5000000);   /* let the run reach steady state */

  const long page = sysconf(_SC_PAGESIZE);
  unsigned char* mem = (unsigned char*)mmap(NULL, (size_t)page * 2,
                                            PROT_READ | PROT_WRITE,
                                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mem == MAP_FAILED) return 2;
  if (mprotect(mem + page, (size_t)page, PROT_NONE) != 0) return 2;
  void* guard = (void*)(mem + page);

  const int fd = open("/dev/null", O_RDONLY);
  errno = 0;
  const int rc = ioctl(fd, 0xDEADBEEFull, guard);   /* not an rknpu submit */
  const int err = errno;
  printf("SELFTEST_WRAPPER foreign_ioctl_rc=%d errno=%d\n", rc, err);
  if (rc != -1) failures++;
  if (err != ENOTTY) {
    printf("SELFTEST_WRAPPER errno_unexpected=%d expected=%d\n", err, ENOTTY);
    failures++;
  }

  const int still_owned = bw_owns(owner);
  printf("SELFTEST_WRAPPER still_owned=%d\n", still_owned);
  if (!still_owned) failures++;

  /* the sampler thread increments this; read it atomically */
  const int sample_before = __atomic_load_n(&g_bw_n, __ATOMIC_ACQUIRE);
  bw_nsleep_ns(3000000);
  const int sample_after = __atomic_load_n(&g_bw_n, __ATOMIC_ACQUIRE);
  printf("SELFTEST_WRAPPER samples_continue=%d before=%d after=%d\n",
         sample_after > sample_before ? 1 : 0, sample_before, sample_after);
  if (sample_after <= sample_before) failures++;

  bw_stop(owner, 0, -1, 0);
  printf("SELFTEST_WRAPPER stopped_after=%d\n", bw_state_is(BW_IDLE) ? 1 : 0);
  if (!bw_state_is(BW_IDLE)) failures++;

  if (fd >= 0) close(fd);
  munmap(mem, (size_t)page * 2);
  printf("SELFTEST_WRAPPER failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}

struct mk_case {
  const char* name;
  uint64_t orig_word;
  int redundant;
  int marker_reg;
  uint32_t marker_value;
  int arm;
  const char* expect;   /* NULL = accept */
};

static int bw_selftest_marker(void) {
  static const struct mk_case cases[] = {
    /* op 0x1001 = DPU target, reg 0x4020 = DPU_DST_BASE_ADDR (plain config) */
    {"accept_marker",  ((uint64_t)0x1001u << 48) | ((uint64_t)0x1234u << 16) | 0x4020u,
     1, 0x6010, 0x00c0ffeeu, MK_ARM_MARKER, NULL},
    {"accept_nop",     ((uint64_t)0x1001u << 48) | ((uint64_t)0x1234u << 16) | 0x4020u,
     1, 0, 0, MK_ARM_NOP, NULL},
    {"arm_invalid",    ((uint64_t)0x1001u << 48) | ((uint64_t)0x1234u << 16) | 0x4020u,
     1, 0x6010, 0x1234u, MK_ARM_ORIGINAL, "arm_invalid"},
    {"slot_is_nop",    0ull, 1, 0x6010, 0x1234u, MK_ARM_MARKER, "slot_is_nop"},
    {"slot_side_ptr",  ((uint64_t)0x1001u << 48) | ((uint64_t)0x0eu << 16) | 0x4004u,
     1, 0x6010, 0x1234u, MK_ARM_MARKER, "slot_word_side_effect"},
    {"slot_side_en",   ((uint64_t)0x6001u << 48) | ((uint64_t)0x0du << 16) | 0x6008u,
     1, 0x6010, 0x1234u, MK_ARM_MARKER, "slot_word_side_effect"},
    {"slot_side_lut",  ((uint64_t)0x1001u << 48) | ((uint64_t)0x1u << 16) | 0x4104u,
     1, 0x6010, 0x1234u, MK_ARM_MARKER, "slot_word_side_effect"},
    {"not_redundant",  ((uint64_t)0x1001u << 48) | ((uint64_t)0x1234u << 16) | 0x4020u,
     0, 0x6010, 0x1234u, MK_ARM_MARKER, "slot_not_redundant"},
    {"nop_not_redundant", ((uint64_t)0x1001u << 48) | ((uint64_t)0x1234u << 16) | 0x4020u,
     0, 0, 0, MK_ARM_NOP, "slot_not_redundant"},
    {"marker_reg_cna", ((uint64_t)0x1001u << 48) | ((uint64_t)0x1234u << 16) | 0x4020u,
     1, 0x1024, 0x1234u, MK_ARM_MARKER, "marker_reg_forbidden"},
    {"marker_reg_ptr", ((uint64_t)0x1001u << 48) | ((uint64_t)0x1234u << 16) | 0x4020u,
     1, 0x6004, 0x1234u, MK_ARM_MARKER, "marker_reg_forbidden"},
    {"marker_reg_en",  ((uint64_t)0x1001u << 48) | ((uint64_t)0x1234u << 16) | 0x4020u,
     1, 0x6008, 0x1234u, MK_ARM_MARKER, "marker_reg_forbidden"},
    {"marker_reg_status", ((uint64_t)0x1001u << 48) | ((uint64_t)0x1234u << 16) | 0x4020u,
     1, 0x6000, 0x1234u, MK_ARM_MARKER, "marker_reg_forbidden"},
    {"marker_reg_unaligned", ((uint64_t)0x1001u << 48) | ((uint64_t)0x1234u << 16) | 0x4020u,
     1, 0x6012, 0x1234u, MK_ARM_MARKER, "marker_reg_forbidden"},
    {"marker_value_zero", ((uint64_t)0x1001u << 48) | ((uint64_t)0x1234u << 16) | 0x4020u,
     1, 0x6010, 0u, MK_ARM_MARKER, "marker_value_zero"},
  };
  int failures = 0;
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    const struct mk_case* c = &cases[i];
    const char* got = marker_check(c->orig_word, c->redundant, c->marker_reg,
                                   c->marker_value, c->arm, 0);
    const int ok = (c->expect == NULL) ? (got == NULL)
                                       : (got != NULL && strcmp(got, c->expect) == 0);
    if (!ok) failures++;
    printf("SELFTEST_MARKER %-22s %s got=%s\n", c->name, ok ? "ok" : "FAIL",
           got ? got : "accept");
  }
  const uint64_t word = marker_word(0x6010, 0x00c0ffeeu);
  const uint64_t want = ((uint64_t)0x4001u << 48) | ((uint64_t)0x00c0ffeeu << 16) | 0x6010u;
  printf("SELFTEST_MARKER word_encode %s value=0x%016llx\n",
         word == want ? "ok" : "FAIL", (unsigned long long)word);
  if (word != want) failures++;
  const uint64_t rdma = marker_word(0x7010, 0x5u);
  const uint64_t rdma_want = ((uint64_t)0x8001u << 48) | ((uint64_t)0x5u << 16) | 0x7010u;
  printf("SELFTEST_MARKER word_encode_rdma %s\n", rdma == rdma_want ? "ok" : "FAIL");
  if (rdma != rdma_want) failures++;
  printf("SELFTEST_MARKER failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}

/* ---------------------------------------------------------------------------
 * Host harness for the marker REWRITE path.
 *
 * marker_check() can be unit tested on its own, but the traversal that actually
 * rewrites a word (task BO -> regcmd IOVA -> word -> restore) could not be run
 * without a board.  This builds the two buffers by hand, registers them as if
 * mmap had been intercepted, and then drives the real marker_apply() /
 * marker_restore() code.
 * ------------------------------------------------------------------------- */
#define MK_BO_TASKS 3
#define MK_BO_WORDS 6
#define MK_BO_TASK_IOVA 0x70000000ull
#define MK_BO_CMD_IOVA  0x71000000ull

static struct rknpu_task mk_bo_tasks[MK_BO_TASKS];
static uint64_t mk_bo_words[MK_BO_TASKS][MK_BO_WORDS];
static uint64_t mk_bo_pristine[MK_BO_TASKS][MK_BO_WORDS];

static uint64_t mk_word_encode(uint16_t op, uint32_t value, uint16_t reg) {
  return ((uint64_t)op << 48) | ((uint64_t)value << 16) | (uint64_t)reg;
}

static int mk_bo_add_record(uint64_t obj_addr, void* host, size_t size, int32_t domain) {
  lock_memory_records();
  for (size_t i = 0; i < MAX_MEMORY_RECORDS; ++i) {
    if (memory_records[i].obj_addr != 0) continue;
    struct memory_record* r = &memory_records[i];
    memset(r, 0, sizeof(*r));
    r->fd = -1;
    r->obj_addr = obj_addr;
    r->size = size;
    r->mapped_address = host;
    r->mapped_length = size;
    r->iommu_domain_id = domain;
    unlock_memory_records();
    return 1;
  }
  unlock_memory_records();
  return 0;
}

static void mk_bo_build(void) {
  const int32_t domain = 7;
  for (int t = 0; t < MK_BO_TASKS; ++t) {
    memset(&mk_bo_tasks[t], 0, sizeof(mk_bo_tasks[t]));
    mk_bo_tasks[t].regcfg_amount = MK_BO_WORDS;
    mk_bo_tasks[t].regcmd_addr = MK_BO_CMD_IOVA + (uint64_t)t * MK_BO_WORDS * 8ull;
    mk_bo_tasks[t].enable_mask = 0x18u;   /* DPU | DPU_RDMA, no PPU */
    /* slot 0 moves the register group: every earlier observation is invalidated */
    mk_bo_words[t][0] = mk_word_encode(0x1001, 0x0000000eu, 0x4004);
    /* slots 1 and 2 write the same (block,reg) with the same value inside the
     * same generation -> slot 2 is genuinely redundant */
    mk_bo_words[t][1] = mk_word_encode(0x1001, 0x1111u + (uint32_t)t, 0x4020);
    mk_bo_words[t][2] = mk_word_encode(0x1001, 0x1111u + (uint32_t)t, 0x4020);
    /* unique per task -> never redundant */
    mk_bo_words[t][3] = mk_word_encode(0x1001, 0x3333u + (uint32_t)t, 0x4028);
    /* identical in every task, but a S_POINTER write sits in between: this is
     * the reviewer's counterexample, it must NOT be treated as redundant */
    mk_bo_words[t][4] = mk_word_encode(0x1001, 0x2222u, 0x4024);
    mk_bo_words[t][5] = 0;   /* OP_NONE */
  }
  memcpy(mk_bo_pristine, mk_bo_words, sizeof(mk_bo_words));
  mk_bo_add_record(MK_BO_TASK_IOVA, mk_bo_tasks, sizeof(mk_bo_tasks), domain);
  mk_bo_add_record(MK_BO_CMD_IOVA, mk_bo_words, sizeof(mk_bo_words), domain);
}

static int mk_bo_run(const char* name, int arm, unsigned task, unsigned slot, int reg,
                     uint32_t value, int expect_applied, const char* expect_reason) {
  memcpy(mk_bo_words, mk_bo_pristine, sizeof(mk_bo_words));
  for (int t = 0; t < MK_BO_TASKS; ++t) mk_bo_tasks[t].enable_mask = 0x18u;
  g_mk_arm = arm;
  g_mk_task = task;
  g_mk_slot = slot;
  g_mk_reg = reg;
  g_mk_value = value;
  g_mk_word_ptr = NULL;
  g_mk_saved = 0;
  g_mk_applied = 0;
  g_mk_expect_set = 0;
  g_mk_fake_sync = 1;          /* the host has no RKNPU mem-sync ioctl */

  struct rknpu_submit s;
  memset(&s, 0, sizeof(s));
  s.flags = 1u;                 /* RKNPU_JOB_PC, blocking */
  s.task_obj_addr = MK_BO_TASK_IOVA;
  s.iommu_domain_id = 7;
  s.task_number = MK_BO_TASKS;
  s.subcore_task[0].task_start = 0;
  s.subcore_task[0].task_number = MK_BO_TASKS;

  marker_apply(-1, &s, 0xBEEFull);
  const int applied_ok = (g_mk_applied == expect_applied);
  const int reason_ok = (expect_reason == NULL) ? (g_mk_reason != NULL)
                                                : (strcmp(g_mk_reason, expect_reason) == 0);
  int content_ok = 1;
  if (expect_applied) {
    const uint64_t expect_word = (arm == MK_ARM_NOP) ? 0ull : marker_word(reg, value);
    content_ok = (mk_bo_words[task][slot] == expect_word) &&
                 (g_mk_orig_word == mk_bo_pristine[task][slot]) &&
                 (g_mk_sync_apply_rc == 0);
    for (int t = 0; t < MK_BO_TASKS; ++t)
      for (int i = 0; i < MK_BO_WORDS; ++i)
        if (!(t == (int)task && i == (int)slot) &&
            mk_bo_words[t][i] != mk_bo_pristine[t][i])
          content_ok = 0;
    if (marker_restore(-1) != 0) content_ok = 0;
    if (memcmp(mk_bo_words, mk_bo_pristine, sizeof(mk_bo_words)) != 0) content_ok = 0;
  } else {
    content_ok = (memcmp(mk_bo_words, mk_bo_pristine, sizeof(mk_bo_words)) == 0);
  }
  const int ok = applied_ok && reason_ok && content_ok;
  printf("SELFTEST_MARKER_BO %-22s %s applied=%d reason=%s%s\n", name,
         ok ? "ok" : "FAIL", g_mk_applied, g_mk_reason,
         content_ok ? "" : " CONTENT_CHANGED");
  g_mk_fake_sync = 0;
  return ok ? 0 : 1;
}

/* Cases that need the submit itself to be hostile. */
static int mk_bo_run_submit(const char* name, int arm, unsigned task, unsigned slot,
                            int reg, uint32_t value, int expect_applied,
                            const char* expect_reason, uint32_t submit_flags,
                            uint32_t enable_mask, int expect_sync_set, uint64_t expect_orig) {
  memcpy(mk_bo_words, mk_bo_pristine, sizeof(mk_bo_words));
  for (int t = 0; t < MK_BO_TASKS; ++t) mk_bo_tasks[t].enable_mask = enable_mask;
  g_mk_arm = arm;
  g_mk_task = task;
  g_mk_slot = slot;
  g_mk_reg = reg;
  g_mk_value = value;
  g_mk_word_ptr = NULL;
  g_mk_saved = 0;
  g_mk_applied = 0;
  g_mk_fake_sync = expect_sync_set;
  g_mk_expect_set = (expect_orig != 0);
  g_mk_expect_orig = expect_orig;

  struct rknpu_submit s;
  memset(&s, 0, sizeof(s));
  s.flags = submit_flags;
  s.task_obj_addr = MK_BO_TASK_IOVA;
  s.iommu_domain_id = 7;
  s.task_number = MK_BO_TASKS;
  s.subcore_task[0].task_start = 0;
  s.subcore_task[0].task_number = MK_BO_TASKS;

  marker_apply(-1, &s, 0xBEEFull);
  const int reason_ok = (g_mk_reason != NULL) &&
                        (strcmp(g_mk_reason, expect_reason) == 0);
  const int applied_ok = (g_mk_applied == expect_applied);
  const int content_ok = (memcmp(mk_bo_words, mk_bo_pristine, sizeof(mk_bo_words)) == 0);
  const int ok = reason_ok && applied_ok && content_ok;
  printf("SELFTEST_MARKER_BO %-22s %s applied=%d reason=%s%s\n", name,
         ok ? "ok" : "FAIL", g_mk_applied, g_mk_reason,
         content_ok ? "" : " CONTENT_CHANGED");
  g_mk_fake_sync = 0;
  g_mk_expect_set = 0;
  return ok ? 0 : 1;
}

static int bw_selftest_marker_bo(void) {
  mk_bo_build();
  int failures = 0;
  failures += mk_bo_run("marker_applied", MK_ARM_MARKER, 1, 2, 0x6010, 0x00c0ffeeu,
                        1, "applied");
  failures += mk_bo_run("nop_applied", MK_ARM_NOP, 1, 2, 0, 0, 1, "applied");
  failures += mk_bo_run("not_redundant", MK_ARM_MARKER, 1, 3, 0x6010, 0x00c0ffeeu,
                        0, "slot_not_redundant");
  /* reviewer counterexample: same value in every task, but a S_POINTER write
   * sits in between, so the second write may initialise the other bank */
  failures += mk_bo_run("cross_task_same_value", MK_ARM_MARKER, 1, 4, 0x6010,
                        0x00c0ffeeu, 0, "slot_not_redundant");
  failures += mk_bo_run("side_effect_slot", MK_ARM_MARKER, 1, 0, 0x6010, 0x00c0ffeeu,
                        0, "slot_word_side_effect");
  failures += mk_bo_run("nop_slot", MK_ARM_MARKER, 1, 5, 0x6010, 0x00c0ffeeu,
                        0, "slot_is_nop");
  failures += mk_bo_run("slot_out_of_range", MK_ARM_MARKER, 1, 99, 0x6010, 0x00c0ffeeu,
                        0, "slot_out_of_range");
  failures += mk_bo_run("task_out_of_range", MK_ARM_MARKER, 99, 2, 0x6010, 0x00c0ffeeu,
                        0, "task_out_of_range");
  failures += mk_bo_run("bad_marker_reg", MK_ARM_MARKER, 1, 2, 0x1024, 0x00c0ffeeu,
                        0, "marker_reg_forbidden");
  failures += mk_bo_run("marker_value_zero", MK_ARM_MARKER, 1, 2, 0x6010, 0u,
                        0, "marker_value_zero");
  /* plan mismatch is caught BEFORE the write */
  failures += mk_bo_run_submit("unexpected_slot_word", MK_ARM_MARKER, 1, 2, 0x6010,
                               0x00c0ffeeu, 0, "unexpected_slot_word", 1u, 0x18u, 1,
                               0xdeadbeefdeadbeefull);
  /* the marker module must not be participating in this submit */
  failures += mk_bo_run_submit("ppu_enabled", MK_ARM_MARKER, 1, 2, 0x6010,
                               0x00c0ffeeu, 0, "ppu_enabled", 1u, 0x38u, 1, 0);
  /* async submit is not supported */
  failures += mk_bo_run_submit("nonblocking_submit", MK_ARM_MARKER, 1, 2, 0x6010,
                               0x00c0ffeeu, 0, "nonblocking_submit", 3u, 0x18u, 1, 0);
  /* a failed BO sync must fail the experiment, not be ignored */
  failures += mk_bo_run_submit("sync_failed_apply", MK_ARM_MARKER, 1, 2, 0x6010,
                               0x00c0ffeeu, 0, "sync_failed_apply", 1u, 0x18u, 0, 0);
  printf("SELFTEST_MARKER_BO failures=%d\n", failures);
  return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  /* Parser-only mode: validate a RAW_OFF string exactly as the env path would.
   * Prints RAW_PARSE ok|reject so the test can assert on the rejection rules. */
  if (arg_long(argc, argv, "--races", 0) != 0) return bw_selftest_races();
  if (arg_long(argc, argv, "--wrapper-probe", 0) != 0) return bw_selftest_wrapper_probe();
  if (arg_long(argc, argv, "--marker-check", 0) != 0) return bw_selftest_marker();
  if (arg_long(argc, argv, "--marker-bo", 0) != 0) return bw_selftest_marker_bo();
  const char* raw_str = arg_str(argc, argv, "--raw-off-str");
  if (raw_str != NULL) {
    int parsed = -1;
    const int experimental = arg_long(argc, argv, "--raw-experimental", 0) != 0;
    const int tier = bw_parse_raw_off(raw_str, &parsed, experimental);
    if (tier != 0) {
      fprintf(stderr, "RAW_PARSE ok value=0x%04x mode=%s\n", parsed,
              bw_raw_mode_name(tier));
      return 0;
    }
    fprintf(stderr, "RAW_PARSE reject\n");
    return 1;
  }
  g_bw_fake = 1;
  /* Read the same environment the ioctl wrapper would read, so the selftest can
   * be driven by run_phase_sweep.sh end to end.  CLI arguments override env. */
  const char* env = NULL;
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_PHASE_NS")) != NULL) g_bw_phase_ns = atol(env);
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_SPIN_NS")) != NULL) g_bw_spin_ns = atol(env);
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_NSLEEP")) != NULL) g_bw_nsleep = atol(env);
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_POLL_NS")) != NULL) g_bw_poll_ns = atol(env);
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_CAP")) != NULL) g_bw_cap = atoi(env);
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_MAX_READS")) != NULL) g_bw_max_reads = atol(env);
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_POWER_CHECK_NS")) != NULL) g_bw_power_check_ns = atol(env);
  int power_file_set = 0;
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_POWER_FILE")) != NULL && env[0] != '\0') {
    snprintf(g_bw_power_path, sizeof(g_bw_power_path), "%s", env);
    power_file_set = 1;
  }
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_RAW_CORE")) != NULL) g_bw_raw_core = atoi(env);
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_TASK_CHECK")) != NULL) g_bw_task_check = atoi(env) != 0;
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_RAW_OFF_EXPERIMENTAL")) != NULL) {
    g_bw_raw_experimental = atoi(env) != 0;
  }
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_RAW_OFF2")) != NULL && env[0] != '\0') {
    int parsed2 = -1;
    if (bw_parse_raw_off(env, &parsed2, g_bw_raw_experimental)) g_bw_raw_off2 = parsed2;
  }
  if ((env = getenv("EPERF_RKNPU_BLOCKWATCH_RAW_OFF")) != NULL) {
    int parsed = -1;
    const int tier = bw_parse_raw_off(env, &parsed, g_bw_raw_experimental);
    if (tier != 0) {
      g_bw_raw_off = parsed;
      g_bw_raw_mode = tier;
      if (tier == 2) {
        fprintf(stderr, "BLOCKWATCH_RAW_EXPERIMENTAL offset=0x%04x\n", parsed);
      }
    } else {
      g_bw_raw_rejected = 1;
      g_bw_raw_off = -1;
      fprintf(stderr, "BLOCKWATCH_RAW_REJECTED value=\"%s\" (allowed: "
              "0x1xxx CNA, 0x3xxx CORE, 0x4xxx DPU, 0x5xxx DPU_RDMA, "
              "0x6xxx PPU, 0x7xxx PPU_RDMA; 4-byte aligned)\n", env);
    }
  }
  /* CLI overrides env only when the flag is actually present */
  long v = 0;
  if (arg_long_opt(argc, argv, "--phase-ns", &v)) g_bw_phase_ns = v;
  if (arg_long_opt(argc, argv, "--poll-ns", &v)) g_bw_poll_ns = v;
  if (arg_long_opt(argc, argv, "--max-reads", &v)) g_bw_max_reads = v;
  if (arg_long_opt(argc, argv, "--cap", &v)) g_bw_cap = (int)v;
  if (arg_long_opt(argc, argv, "--edge", &v)) g_bw_edge = v != 0;
  if (arg_long_opt(argc, argv, "--raw-off", &v)) {
    /* same three tiers as the env path, so the reported mode is never a lie */
    if (bw_raw_offset_forbidden((int)v)) {
      g_bw_raw_off = -1;
      g_bw_raw_mode = 0;
    } else {
      g_bw_raw_off = (int)v;
      g_bw_raw_mode = bw_raw_offset_vetted((int)v) ? 1
                    : (bw_raw_offset_irq_status((int)v) ? 3 : 2);
    }
  }
  if (arg_long_opt(argc, argv, "--power-check-ns", &v)) g_bw_power_check_ns = v;
  if (arg_long_opt(argc, argv, "--fake-busy-after-ns", &v)) g_bw_fake_busy_after_ns = v;
  if (arg_long_opt(argc, argv, "--fake-task", &v)) g_bw_fake_task = v;
  if (arg_long_opt(argc, argv, "--fake-bank-lag", &v)) g_bw_fake_bank_lag = v;
  if (arg_long_opt(argc, argv, "--task-check", &v)) g_bw_task_check = v != 0;
  const char* power_file = arg_str(argc, argv, "--power-file");
  if (power_file != NULL) {
    snprintf(g_bw_power_path, sizeof(g_bw_power_path), "%s", power_file);
    power_file_set = 1;
  }
  /* Host mode has no debugfs power node: only guard when a file was given. */
  if (!power_file_set) g_bw_power_check_ns = 0;
  const long run_ms = arg_long(argc, argv, "--run-ms", 50);
  const long arm_delay_us = arg_long(argc, argv, "--arm-delay-us", 2000);
  const long foreign_first = arg_long(argc, argv, "--arm-foreign-first", 0);
  const long foreign_stop = arg_long(argc, argv, "--stop-foreign", 0);
  static uint64_t owner_cookie = 0xA5A5A5A5ull;
  if (bw_start(owner_cookie) != 0) {
    fprintf(stderr, "SELFTEST_START_FAILED\n");
    return 2;
  }
  struct timespec d;
  d.tv_sec = arm_delay_us / 1000000L;
  d.tv_nsec = (arm_delay_us % 1000000L) * 1000L;
  nanosleep(&d, NULL);
  if (foreign_first) {
    /* A different ioctl must not be able to move the time origin. */
    bw_arm(owner_cookie ^ 0x5A5A5A5Aull);
    d.tv_sec = 0;
    d.tv_nsec = 15000000L;
    nanosleep(&d, NULL);
  }
  bw_arm(owner_cookie);
  if (arg_long(argc, argv, "--note-ioctl", 0) != 0) {
    /* a real ioctl that entered 2 us after the submit entry and returned at 60 us */
    const uint64_t enter = __atomic_load_n(&g_bw_submit_enter_ns, __ATOMIC_ACQUIRE);
    bw_note_ioctl(enter + 2000, enter + 60000);
  }
  d.tv_sec = run_ms / 1000L;
  d.tv_nsec = (run_ms % 1000L) * 1000000L;
  nanosleep(&d, NULL);
  if (foreign_stop) {
    /* A foreign call must not be able to tear the run down. */
    bw_stop(owner_cookie ^ 0x5A5A5A5Aull, 0, -1, 0);
  }
  bw_stop(owner_cookie, 0, -1, 0);
  return 0;
}
#endif
