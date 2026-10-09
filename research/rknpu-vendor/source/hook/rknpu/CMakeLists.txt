add_library(rknpu_ioctl_hook SHARED
  rknpu_ioctl_hook.c
)

find_package(Threads REQUIRED)
target_link_libraries(rknpu_ioctl_hook PRIVATE dl Threads::Threads)

set_target_properties(rknpu_ioctl_hook PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME eperf_hook_rknpu_ioctl
)

target_compile_options(rknpu_ioctl_hook PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
