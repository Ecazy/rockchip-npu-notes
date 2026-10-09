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
static int g_probe_seen;
static int g_probe_saved_tn;
static struct rknpu_subcore_task g_probe_saved_sc[5];
static int g_probe_done;
static int g_probe_orig_tn;
static void* g_probe_arg;            /* identifies the truncated ioctl (multi-thread safe) */
static uint32_t g_probe_newsum;      /* truncated total task_number, for identity check */
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
  size_t bytes = entries * sizeof(uint64_t);
  if (bytes == 0 || bytes > available) {
    bytes = available < limit ? available : limit;
  } else if (bytes > limit) {
    bytes = limit;
  }
  const unsigned char* data = (const unsigned char*)host;
  char record[REGCMD_HEX_BUDGET + 512];
  int length = snprintf(
      record, sizeof(record),
      "{\"schema_version\":1,\"event\":\"regcmd\",\"sequence\":%llu,"
      "\"task_index\":%zu,\"op_idx\":%u,\"regcfg_amount\":%u,"
      "\"bytes\":%zu,\"hex\":\"",
      sequence, task_index, capture->op_idx, capture->regcfg_amount, bytes);
  if (length <= 0 || (size_t)length >= sizeof(record)) {
    return;
  }
  size_t used = (size_t)length;
  static const char digits[] = "0123456789abcdef";
  for (size_t index = 0; index < bytes; ++index) {
    if (used + 3 >= sizeof(record)) {
      break;
    }
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
 * main thread is blocked inside the real submit ioctl.  SAFETY: power must be
 * on; a periodic guard aborts sampling if the domain drops. */
struct bw_core {
  uint32_t ts, cna, core, dpu;
};
struct bw_sample {
  uint64_t t;
  struct bw_core c[3]; /* core0 @0xfdab0000, core1 @0xfdac0000, core2 @0xfdad0000 */
};
static int g_bw_enabled, g_bw_min_eff = 4000, g_bw_nth = 1, g_bw_seen;
static int g_bw_op_idx = -1, g_bw_require_all;
static volatile int g_bw_run;
static int g_bw_n, g_bw_cap = 400000, g_bw_fd = -1;
static volatile uint32_t* g_bw_regs;
static struct bw_sample* g_bw_buf;
static pthread_t g_bw_th;
static int g_bw_active;
static int g_bw_all_cores;
static int g_bw_minimal;
static long g_bw_nsleep;

static int bw_power_on(void) {
  FILE* f = fopen("/sys/kernel/debug/rknpu/power", "r");
  if (f == NULL) return -1;
  char b[8] = {0};
  int ok = (fgets(b, sizeof(b), f) != NULL) && strncmp(b, "on", 2) == 0;
  fclose(f);
  return ok ? 1 : 0;
}

static void* bw_thread(void* arg) {
  (void)arg;
  while (__atomic_load_n(&g_bw_run, __ATOMIC_RELAXED)) {
    int i = g_bw_n;
    if (i >= g_bw_cap) break;
    struct bw_sample* s = &g_bw_buf[i];
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    s->t = (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
    int nc = g_bw_all_cores ? 3 : 1;
    for (int k = 0; k < nc; ++k) {
      const volatile uint32_t* r = g_bw_regs + (size_t)k * (0x10000 / 4);
      if (g_bw_minimal) {
        s->c[k].ts = r[0x003c / 4];
        s->c[k].cna = s->c[k].core = s->c[k].dpu = 0;
      } else {
        s->c[k].ts = r[0x003c / 4];
        s->c[k].cna = r[0x1000 / 4];
        s->c[k].core = r[0x3000 / 4];
        s->c[k].dpu = r[0x4000 / 4];
      }
    }
    g_bw_n = i + 1;
    if (g_bw_nsleep > 0) {
      struct timespec d;
      d.tv_sec = g_bw_nsleep / 1000000000L;
      d.tv_nsec = g_bw_nsleep % 1000000000L;
      nanosleep(&d, NULL);
    }
    /* no per-sample debugfs read: the blocking submit ioctl holds the power
     * reference, so the domain cannot drop during this window. */
  }
  return NULL;
}

static int bw_start(void) {
  if (bw_power_on() != 1) return -1; /* never touch MMIO with the domain off */
  g_bw_fd = open("/dev/mem", O_RDONLY | O_SYNC);
  if (g_bw_fd < 0) return -1;
  void* m = mmap(NULL, 0x30000, PROT_READ, MAP_SHARED, g_bw_fd, (off_t)0xfdab0000);
  if (m == MAP_FAILED) { close(g_bw_fd); g_bw_fd = -1; return -1; }
  g_bw_regs = (volatile uint32_t*)m;   /* whole 3-core window starts here */
  g_bw_buf = (struct bw_sample*)malloc((size_t)g_bw_cap * sizeof(*g_bw_buf));
  if (g_bw_buf == NULL) { munmap(m, 0x30000); close(g_bw_fd); g_bw_fd = -1; return -1; }
  g_bw_n = 0;
  __atomic_store_n(&g_bw_run, 1, __ATOMIC_RELAXED);
  if (pthread_create(&g_bw_th, NULL, bw_thread, NULL) != 0) {
    __atomic_store_n(&g_bw_run, 0, __ATOMIC_RELAXED);
    free(g_bw_buf); munmap(m, 0x30000); close(g_bw_fd); g_bw_fd = -1; return -1;
  }
  g_bw_active = 1;
  /* handshake: ensure the sampler is running before the submit starts */
  while (__atomic_load_n(&g_bw_n, __ATOMIC_RELAXED) == 0) sched_yield();
  return 0;
}

static void bw_stop(int result, long long hw_elapse, long long wall_ns) {
  if (!g_bw_active) return;
  __atomic_store_n(&g_bw_run, 0, __ATOMIC_RELAXED);
  pthread_join(g_bw_th, NULL);
  const char* path = getenv("EPERF_RKNPU_BLOCKWATCH_FILE");
  if (path != NULL) {
    FILE* f = fopen(path, "w");
    if (f != NULL) {
      const int nc = g_bw_all_cores ? 3 : 1;
      fprintf(f, "t_ns");
      for (int k = 0; k < nc; ++k) fprintf(f, ",c%d_ts,c%d_cna,c%d_core,c%d_dpu", k, k, k, k);
      fprintf(f, "\n");
      for (int i = 0; i < g_bw_n; ++i) {
        const struct bw_sample* s = &g_bw_buf[i];
        fprintf(f, "%llu", (unsigned long long)s->t);
        for (int k = 0; k < nc; ++k)
          fprintf(f, ",0x%08x,0x%08x,0x%08x,0x%08x", s->c[k].ts, s->c[k].cna,
                  s->c[k].core, s->c[k].dpu);
        fprintf(f, "\n");
      }
      fclose(f);
    }
  }
  {
    uint32_t last = g_bw_n > 0 ? g_bw_buf[g_bw_n - 1].c[0].ts : 0;
    fprintf(stderr,
            "BLOCKWATCH_META result=%d hw_elapse_ns=%lld wall_ns=%lld samples=%d cap=%d "
            "overflow=%d c0_counter0=%u c0_counter1=%u all_cores=%d probe=%d dedup=%d\n",
            result, hw_elapse, wall_ns, g_bw_n, g_bw_cap,
            g_bw_n >= g_bw_cap ? 1 : 0, last & 0xfffu, (last >> 16) & 0xfffu,
            g_bw_all_cores, getenv("EPERF_RKNPU_PROBE") != NULL,
            getenv("EPERF_RKNPU_DEDUP") != NULL);
  }
  fflush(stderr);
  munmap((void*)g_bw_regs, 0x30000);
  close(g_bw_fd); g_bw_fd = -1; g_bw_regs = NULL;
  free(g_bw_buf); g_bw_buf = NULL; g_bw_active = 0;
}

/* Duplicate-write elimination (causal ablation): replace a non-side-effect
 * NPUOP whose value equals the last value written to the same (block,reg) with
 * OP_NONE, preserving regcmd length and task count.  Side-effect registers
 * (*_S_POINTER, OP_ENABLE, LUT_ACCESS_*, PC/interrupt) are never touched. */
static int g_ded_enabled, g_ded_min_eff = 4000, g_ded_nth = 1, g_ded_seen, g_ded_done, g_ded_exit, g_ded_reported;
static uint32_t g_ded_last[8 * 0x10000];
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
  memset(g_ded_last, 0, sizeof(g_ded_last));
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
      if (task->regcfg_amount == 0) continue;
      if (g_ded_per_task) memset(g_ded_last, 0, sizeof(g_ded_last));
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
        if (op == 0 || ded_side_effect(op, reg)) continue;
        const unsigned idx = (unsigned)ded_block_idx(op) * 0x10000u + reg;
        if (g_ded_last[idx] == val + 1u) {
          ded_save(&w[i], u);
          w[i] = 0; /* OP_NONE */
          g_ded_repl++;
          touched = 1;
        } else {
          g_ded_last[idx] = val + 1u;
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

int ioctl(int fd, unsigned long request, ...) {
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
    }
    if (g_probe_enabled && g_probe_L > 0) {
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
  if (submit && argument != NULL && !g_bw_active) {
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
        if (g_bw_seen == g_bw_nth && bw_start() == 0) {
          fprintf(stderr, "BLOCKWATCH start: task_number=%u scsum=%u\n",
                  s->task_number, scsum);
          fflush(stderr);
        }
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
    if (g_probe_exit && result == 0) _exit(0);
    if (!g_probe_exit) { /* restore original submit params for the runtime */
      pa->task_number = (uint32_t)g_probe_orig_tn;
      for (int i = 0; i < 5; ++i) pa->subcore_task[i] = g_probe_saved_sc[i];
    }
  }
  if (g_bw_active) {
    const struct rknpu_submit* pba = (const struct rknpu_submit*)argument;
    bw_stop(result, pba ? (long long)pba->hw_elapse_time : -1, end_ns - start_ns);
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
