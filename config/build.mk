CXX := hipcc

OUTPUT := librccl-net-tcpx.so
SONAME := $(OUTPUT)

LDFLAGS := -L$(RCCL_ROOT)/lib
LDLIBS := -lamdhip64 -lhsa-runtime64

CPPFLAGS := -D__HIP_PLATFORM_AMD__ -I$(RCCL_ROOT)
