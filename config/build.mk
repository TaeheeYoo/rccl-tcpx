OUTPUT := librccl-net-tcpx.so
SONAME := $(OUTPUT)

LDFLAGS := -lamdhip64 -lhsa-runtime64 -lynl

CPPFLAGS := -D__HIP_PLATFORM_AMD__
