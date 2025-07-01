#ifndef TCPX_H__ 
#define TCPX_H__

#define __iovec_defined
#include <linux/uio.h>		// struct iovec
#include <sys/socket.h>		// CMSG_SPACE

#define __hidden __attribute__ ((visibility("hidden")))

#define NCCL_PLUGIN_MAX_RECVS   1
#define IF_NAME_SIZE        	16
#define MAX_IFS                 16
#define CTRL_DATA_LEN		CMSG_SPACE(sizeof(struct iovec) * 10000)

#define DMABUF_SIZE	4096
#define N_QUEUES	1

#endif  // THIRD_PARTY_GPUS_RCCL_TCPX_PLUGIN_UTIL_H_
