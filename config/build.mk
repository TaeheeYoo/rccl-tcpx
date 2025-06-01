OUTPUT := librccl-net-tcpx.so
SONAME := $(OUTPUT)

LDFLAGS := -L /root/tools/net/ynl/lib -L /root/tools/net/ynl			\
		   -L /opt/rocm/lib												\
		   -lamdhip64 -lhsa-runtime64 -lynl								\

CPPFLAGS := -I /usr/include -I /usr/src/$(shell uname -r)/include 		\
			-I /root/tools/net/ynl/generated -I /root/tools/net/ynl/lib	\
			-I /opt/rocm/include										\
			-D__HIP_PLATFORM_AMD__
