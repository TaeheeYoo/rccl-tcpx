PLUGIN_SO:=librccl-net-tcpx.so

default: $(PLUGIN_SO)

$(PLUGIN_SO): tcpx.c
	      $(CC) -I. \
		-I /usr/include \
		-I /usr/src/`uname -r`/include \
		-I /root/tools/net/ynl/generated \
		-I /root/tools/net/ynl/lib \
		-I /opt/rocm/include \
		-fPIC -shared \
		-L. \
		-L /root/tools/net/ynl/lib \
		-L /root/tools/net/ynl \
		-L /opt/rocm/lib \
		-lamdhip64 \
		-lhsa-runtime64 \
		-lynl \
		-D__HIP_PLATFORM_AMD__ \
		-o $@ -Wl,-soname,$(PLUGIN_SO) $^

clean:
	rm -f $(PLUGIN_SO)
