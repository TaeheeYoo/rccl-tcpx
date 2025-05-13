lib_LIBRARIES = librccl-tcpx.a

HIPCC=$(HIP_PATH)/bin/hipcc

CXX = hipcc
CC = hipcc

librccl_tcpx_a_SOURCES = tcpx.cpp tcpx.hpp

librccl_tcpx_CXXFLAGS = -I. \
		      -I /usr/include \
		      -I /usr/src/`uname -r`/include \
		      -I /root/tools/net/ynl/generated \
		      -I /root/tools/net/ynl/lib \
		      -I /opt/rocm/include \
		      --emit-static-lib -fPIC \
		      -L. \
		      -L /root/tools/net/ynl/lib \
		      -L /root/tools/net/ynl \
		      -L /opt/rocm/lib \
		      -lamdhip64 \
		      -lhsa-runtime64 \
		      -lynl \
		      -D__HIP_PLATFORM_AMD__
