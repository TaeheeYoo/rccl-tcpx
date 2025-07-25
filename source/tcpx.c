#include "tcpx.h"

#include "net.h"
#include "net_v9.h"
#include "net_v8.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <unistd.h>

#include <arpa/inet.h>
#include <net/if.h>
#include <ifaddrs.h>

#include "logger.h"

#include "ncdevmem.h"

#include "amdgpu_memory_provider.h"
#include "amdgpu_dmabuf_provider.h"

int max_requests = NCCL_NET_MAX_REQUESTS;
int ncclNetIfs = 0;

struct amdgpu_memory_provider *provider = &amdgpu_dmabuf_provider;
struct amdgpu_memory_buffer *dma_buffer;
struct ncdevmem *ncdevmem;

enum nccl_socket_ops {
	NCCL_SOCKET_SEND = 0,
	NCCL_SOCKET_RECV = 1
};

struct nccl_net_socket_request {
	enum nccl_socket_ops op;
	void *data;
	int size;
	int used;

	struct nccl_net_socket_comm *comm;
};

struct nccl_net_socket_comm {
	int fd;
	int num_socks;
	int num_threads;
	int dev;

	struct nccl_net_socket_request requests[NCCL_NET_MAX_REQUESTS];
};

struct nccl_net_socket_listen_comm {
	int fd;
	int num_socks;
	int num_threads;
	int dev;
};

union socket_address {
	struct sockaddr sa;
	struct sockaddr_in sin;
	struct sockaddr_in6 sin6;
};

struct nccl_net_socket_handle {
	union socket_address connect_addr;
	int num_socks;
	int num_threads;
};

struct tcpx_dev {
	union socket_address addr;
	char dev_name[IF_NAME_SIZE];
	char* pci_path;
};

struct tcpx_memory_handle {
	void* uptr;
	void* ptr;
	int mem_type;
};

static struct tcpx_dev tcpx_devs[MAX_IFS];

/* TODO NCCL_TCPX_SUBNET */
__hidden ncclResult_t tcpx_init(ncclDebugLogger_t logFunction)
{
	char* ifs = getenv("NCCL_TCPX_IFNAMES");
	struct ifaddrs *ifaddr;

	if (!logger_initialize()) {
		logFunction(
			NCCL_LOG_ABORT, NCCL_INIT, __FILE__, __LINE__,
	      		"failed to logger_initialize(): %s", strerror(errno)
		);
		return ncclInternalError;
	}

	if (!ifs) {
		log(ERRN, "interfaces are not defined");
		return ncclInternalError;
	}

	if (getifaddrs(&ifaddr) == -1) {
		log(PERRN, "failed to getifaddrs(): ");
		return ncclInternalError;
	}

	for (char *token = strtok(ifs, ",");
	           token != NULL;
	  	   token = strtok(NULL, ","))
	{
		for (struct ifaddrs *ifa = ifaddr;
		     ifa != NULL;
		     ifa = ifa->ifa_next)
		{
			if (!ifa->ifa_addr)
				continue;

			if (strcmp(ifa->ifa_name, token))
				continue;

			if (ifa->ifa_addr->sa_family != AF_INET)
				continue;

			memset(&tcpx_devs[ncclNetIfs].addr, 0x00,
	  		       sizeof(union socket_address));
			memcpy(&tcpx_devs[ncclNetIfs].addr.sin, ifa->ifa_addr,
			       sizeof(struct sockaddr_in));
			strcpy(tcpx_devs[ncclNetIfs].dev_name, token);

			ncclNetIfs++;
			break;
		}

		if (ncclNetIfs >= MAX_IFS) {
			log(WARN, "Maximum number of interfaces reached.");
			break;
		}
	}

	freeifaddrs(ifaddr);

	if (ncclNetIfs == 0) {
		log(ERRN, "failed to find any matching interfaces");
		return ncclSystemError;
	}

	/*
	dma_buffer = provider->alloc(DMABUF_SIZE * getpagesize());
	if (dma_buffer == NULL) {
		log(ERRN, "failed to allocate dma-buffer");
		return ncclSystemError;
	}

	ncdevmem = ncdevmem_setup(
		if_nametoindex(tcpx_devs[0].dev_name),
		N_QUEUES,
		dma_buffer->fd
	);
	if (ncdevmem == NULL) {
		provider->free(dma_buffer);
		log(ERRN, "failed to setup ncdevmem");
		return ncclSystemError;
	}
	*/

	log(INFO, "tcpx_init() complete: %d interface(s)", ncclNetIfs);
	for (int i = 0; i < ncclNetIfs; i++) {
		log(INFO, "\tname: %s", tcpx_devs[i].dev_name);
		log(INFO, "\tpci_path: %s", tcpx_devs[i].pci_path);
		do {
			struct sockaddr_in *addr_in = &tcpx_devs[i].addr.sin;
			char ip_str[INET6_ADDRSTRLEN];
			int port;

			inet_ntop(AF_INET, &addr_in->sin_addr,
	     			  ip_str, sizeof(ip_str));
			port = ntohs(addr_in->sin_port);

			log(INFO, "\taddr: %s:%d", ip_str, port);
		} while (false);
	}

	return ncclSuccess;
}

__hidden ncclResult_t tcpx_devices(int* ndev)
{
	/* interface index? */
	*ndev = ncclNetIfs;

	log(INFO, "tcpx_devices() complete: %d device(s)", ncclNetIfs);

	return ncclSuccess;
}

__hidden ncclResult_t tcpx_listen(int dev, void *opaque_handle,
				  void **listen_comm)
{
	struct nccl_net_socket_listen_comm *comm;
	int family, salen, err, sockfd, opt = 1;
	struct nccl_net_socket_handle* handle;
	ncclResult_t retval;

	handle = (struct nccl_net_socket_handle *) opaque_handle;

	if (dev < 0 || dev >= ncclNetIfs) {
		log(WARN, "tcpx_listen dev=%d ncclNetIfs=%d",
      			   dev, ncclNetIfs);
		retval = ncclInternalError;
		goto RETURN_ERROR;
	}

	comm = calloc(1, sizeof(struct nccl_net_socket_listen_comm));
	if (!comm) {
		log(PWARN, "Failed to allocate memory: ");
		retval = ncclInternalError;
		goto RETURN_ERROR;
	}

	memcpy((void *)&handle->connect_addr, &tcpx_devs[dev].addr,
	       sizeof(handle->connect_addr));
	family = handle->connect_addr.sa.sa_family;
	salen = (family == AF_INET) ? sizeof(struct sockaddr_in) :
				      sizeof(struct sockaddr_in6);

	/* Create socket and bind it to a port */
	sockfd = socket(family, SOCK_STREAM, 0);
	if (sockfd == -1) {
		log(PWARN, "Failed to socket(): ");
		retval = ncclSystemError;
		goto FREE_COMM;
	}

	err = setsockopt(sockfd, SOL_SOCKET,
			 SO_REUSEADDR | SO_REUSEPORT,
			 &opt, sizeof(opt));
	if (err) {
		log(PWARN, "Failed to setsockopt: ");
		retval = ncclSystemError;
		goto CLOSE_SOCKFD;
	}

	err = bind(sockfd, &handle->connect_addr.sa, salen);
	if (err) {
		log(PWARN, "Failed to bind(): ");
		retval = ncclSystemError;
		goto CLOSE_SOCKFD;
	}

	/* Get the assigned Port */
	socklen_t size = salen;
	err = getsockname(sockfd, &handle->connect_addr.sa, &size);
	if (err) {
		log(PWARN, "Failed to getsockname(): ");
		retval = ncclSystemError;
		goto CLOSE_SOCKFD;
	}

	/* Put the socket in listen mode
	 * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
	 */
	err = listen(sockfd, 16384);
	if (err) {
		log(PWARN, "Failed to listen(): ");
		retval = ncclSystemError;
		goto CLOSE_SOCKFD;
	}

	comm->fd = sockfd;
	comm->num_socks = 0;
	comm->num_threads = 0;
	comm->dev = dev;

	handle->num_socks = comm->num_socks;
	handle->num_threads = comm->num_threads;

	*listen_comm = comm;

	log(INFO, "tcpx_listen() complete:");
	log(INFO, "\tdevice: %d", dev);
	log(INFO, "\tcomm->fd: %d", comm->fd);
	log(INFO, "\thandle->num_socks: %d", comm->num_socks);
	log(INFO, "\thandle->num_threads: %d", comm->num_threads);
	do {
		struct sockaddr_in *addr_in = &handle->connect_addr.sin;
		char ip_str[INET6_ADDRSTRLEN];
		int port;
	
		inet_ntop(AF_INET, &addr_in->sin_addr, ip_str, sizeof(ip_str));
		port = ntohs(addr_in->sin_port);

		log(INFO, "\thandle->addr: %s:%d", ip_str, port);
	} while (false);

	return ncclSuccess;

CLOSE_SOCKFD:	close(sockfd);
FREE_COMM:	free(comm);
RETURN_ERROR:	return retval;
}

__hidden ncclResult_t tcpx_connect(int dev, void* opaqueHandle,
				   void** sendComm,
				   ncclNetDeviceHandle_t** sendDevComm)
{
	struct nccl_net_socket_handle *handle = opaqueHandle;
	struct nccl_net_socket_comm *comm;
	ncclResult_t retval;
	size_t addrlen;

	comm = calloc(1, sizeof(struct nccl_net_socket_comm));
	if (comm == NULL) {
		log(PWARN, "Failed to calloc(): ");
		retval = ncclInternalError;
		goto RETURN_ERROR;
	}

	comm->fd = socket(handle->connect_addr.sa.sa_family, SOCK_STREAM, 0);
	if (comm->fd == -1) {
		log(PWARN, "Failed to socket(): ");
		retval = ncclSystemError;
		goto FREE_COMM;
	}

	addrlen = (handle->connect_addr.sa.sa_family == AF_INET) ?
		  sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
	if (connect(comm->fd, &handle->connect_addr.sa, addrlen) == -1) {
		log(PWARN, "Failed to connect(): ");
		retval = ncclInternalError;
		goto CLOSE_COMM_FD;
	}

	*sendComm = comm;

	log(INFO, "tcpx_connect() complete:");
	log(INFO, "\tcomm->fd: %d", comm->fd);
	do {
		struct sockaddr_in *addr_in = &handle->connect_addr.sin;
		char ip_str[INET6_ADDRSTRLEN];
		int port;
		
		inet_ntop(AF_INET, &addr_in->sin_addr, ip_str, sizeof(ip_str));
		port = ntohs(addr_in->sin_port);

		log(INFO, "\thandle->addr: %s:%d", ip_str, port);
	} while (false);

	return ncclSuccess;

CLOSE_COMM_FD:	close(comm->fd);
FREE_COMM:	free(comm);
RETURN_ERROR:	return retval;
}

__hidden ncclResult_t tcpx_accept(void* listenComm, void** recvComm,
				  ncclNetDeviceHandle_t** recvDevComm)
{
	struct nccl_net_socket_listen_comm *lcomm = listenComm;
	struct nccl_net_socket_comm *rcomm;
	ncclResult_t retval;

	rcomm = calloc(1, sizeof(struct nccl_net_socket_comm));
	if (rcomm == NULL) {
		log(PWARN, "Failed to calloc(): ");
		retval = ncclInternalError;
		goto RETURN_ERROR;
	}

	rcomm->num_socks = lcomm->num_socks;
	rcomm->num_threads = lcomm->num_threads;
	rcomm->dev = lcomm->dev;

	rcomm->fd = accept(lcomm->fd, NULL, 0);
	if (rcomm->fd == -1) {
		log(PWARN, "Failed to accept(): ");
		retval = ncclInternalError;
		goto FREE_RCOMM;
	}

	*recvComm = rcomm;

	log(INFO, "tcpx_accept() complete: lcomm->fd: %d", lcomm->fd);
	log(INFO, "\trcomm->fd: %d", rcomm->fd);
	log(INFO, "\trcomm->dev: %d", rcomm->dev);
	log(INFO, "\trcomm->num_socks: %d", rcomm->num_socks);
	log(INFO, "\trcomm->num_threads: %d", rcomm->num_threads);

	return ncclSuccess;

FREE_RCOMM:	free(rcomm);
RETURN_ERROR:	return retval;
}

__hidden ncclResult_t tcpx_reg_mr(void* internal_comm, void* data, size_t size,
				  int type, void** mhandle)
{
	struct nccl_net_socket_comm *comm = internal_comm;
	struct tcpx_memory_handle *handle;

	handle = calloc(1, sizeof(struct tcpx_memory_handle));
	if (handle == NULL) {
		log(PERRN, "failed to calloc(): ");
		return ncclInternalError;
	}	
	
	handle->mem_type = type;

	switch (handle->mem_type) {
	case NCCL_PTR_HOST:
		handle->uptr = data;
		handle->ptr = data;
		break;

	case NCCL_PTR_CUDA:
	default:
		log(PWARN, "unknown mem_type: %d", handle->mem_type);
		return ncclInternalError;
	}

	*mhandle = handle;

	log(INFO, "tcpx_reg_mr() complete: ");
	log(INFO, "\tcomm->fd: %d", comm->fd);
	log(INFO, "\tcomm->dev: %d", comm->dev);
	log(INFO, "\thandle->type: %d", handle->mem_type);
	log(INFO, "\thandle->ptr: %p", handle->ptr); 
	log(INFO, "\thandle->uptr: %zu", handle->uptr);
 
	return ncclSuccess;
}

__hidden ncclResult_t tcpx_reg_mr_dmabuf(void* comm, void* data, size_t size,
					int type, uint64_t offset, int fd,
					void** mhandle)
{
	log(WARN, "tcpx_reg_mr_dmabuf() is not implemented yet.");
	return ncclSuccess;
}

__hidden ncclResult_t tcpx_dereg_mr(void* internal_comm, void* mhandle)
{
	struct nccl_net_socket_comm *comm = internal_comm;
	struct tcpx_memory_handle *handle = mhandle;

	log(INFO, "tcpx_dereg_mr() complete: ");
	log(INFO, "\tcomm->fd: %d", comm->fd);
	log(INFO, "\tcomm->dev: %d", comm->dev);
	log(INFO, "\thandle->type: %d", handle->mem_type);
	log(INFO, "\thandle->ptr: %p", handle->ptr); 
	log(INFO, "\thandle->uptr: %zu", handle->uptr);

	free(mhandle);

	return ncclSuccess;
}

__hidden ncclResult_t tcpx_isend(void* sendComm, void* data, size_t size,
				 int tag, void* mhandle, void** request)
{
	struct nccl_net_socket_comm *comm = sendComm;
	struct nccl_net_socket_request *req;

	for (int i = 0; i < NCCL_NET_MAX_REQUESTS; i++) {
		req = comm->requests + i;
		if (req->used != 0)
			continue;

		req->op = NCCL_SOCKET_SEND;
		req->data = data;
		req->size = size;
		req->used = 1;
		req->comm = comm;

		*request = req;

		log(INFO, "tcpx_isend() complete:");
		log(INFO, "\tcomm->dev: %d", comm->dev);
		log(INFO, "\tcomm->fd: %d", comm->fd);
		log(INFO, "\tcomm->requests index: %d", i);

		return ncclSuccess;
	}

	log(WARN, "failed to tcpx_isend(): request queue is full");

	return ncclInternalError;
}

__hidden ncclResult_t tcpx_irecv(void* recvComm, int n, void** data,
				 size_t* sizes, int* tags, void** mhandles,
				 void** request)
{
	struct nccl_net_socket_comm *comm = recvComm;
	struct nccl_net_socket_request *req;

	for (int i = 0; i < NCCL_NET_MAX_REQUESTS; i++) {
		req = comm->requests + i;
		if (req->used != 0)
			continue;

		req->op = NCCL_SOCKET_RECV;
		req->data = data[0];
		req->size = sizes[0];
		req->used = 1;
		req->comm = comm;

		*request = req;

		log(INFO, "tcpx_irecv() complete:");
		log(INFO, "\tcomm->dev: %d", comm->dev);
		log(INFO, "\tcomm->fd: %d", comm->fd);
		log(INFO, "\tcomm->requests index: %d", i);

		return ncclSuccess;
	}

	log(WARN, "failed to tcpx_irecv(): request queue is full");

	return ncclInternalError;
}

__hidden ncclResult_t tcpx_iflush(void* recvComm, int n, void** data,
				  int* sizes, void** mhandles, void** request)
{
	log(WARN, "tcpx_iflush() is not implemented yet.");
	return ncclSuccess;
}

int recv_all(int fd, void *data, int size)
{
	int recv_len = 0;

	while (recv_len < size) {
		int len = recv(fd, data + recv_len, size - recv_len, 0);
		if (len <= 0)
			return len;

		recv_len += len;
	}

	return recv_len;
}

int send_all(int fd, void *data, int size)
{
	int send_len = 0;

	while (send_len < size) {
		int len = send(fd, data + send_len, size - send_len, 0);
		if (len <= 0)
			return len;

		send_len += len;
	}

	return send_len;
}

int recv_dma(int fd, void *data, int size)
{
	struct msghdr msg;
	struct iovec iovec;

	char ctrl_data[CTRL_DATA_LEN];
	int ret;

	log(INFO, "recv_dma(%d, %p, %d):", fd, data, size);

	msg.msg_control = ctrl_data;
	msg.msg_controllen = CTRL_DATA_LEN;

	msg.msg_iov = &iovec;
	msg.msg_iovlen = 1;

	iovec.iov_base = data;
	iovec.iov_len = size;

	ret = recvmsg(fd, &msg, MSG_SOCK_DEVMEM);
	if (ret == -1) return -1;
	if (ret ==  0) return  0;

	for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
      	     cmsg; cmsg = CMSG_NXTHDR(&msg, cmsg))
	{
		struct dmabuf_cmsg *dmabuf_cmsg;
		size_t offset, size;

		dmabuf_cmsg = (struct dmabuf_cmsg *) CMSG_DATA(cmsg);

		log(INFO, "\tcmsg->type: %d", cmsg->cmsg_type);
		log(INFO, "\tcmsg->len: %d", cmsg->cmsg_len);
		log(INFO, "\tcmsg->level: %d", cmsg->cmsg_level);

		if (cmsg->cmsg_type == SCM_DEVMEM_DMABUF) {
			offset = dmabuf_cmsg->frag_offset;
			size = dmabuf_cmsg->frag_size;

			log(INFO, "\tdmabuf_cmsg->frag_offset: %d", offset);
			log(INFO, "\tdmabuf_cmsg->frag_size: %d", size);

			provider->memmove_to(dma_buffer, data, offset, size);
		} else if (cmsg->cmsg_type == SCM_DEVMEM_LINEAR) {
			size = dmabuf_cmsg->frag_size;

			log(INFO, "\tdmabuf_cmsg->frag_size: %d", size);
		}
	}

	return 0;
}

__hidden ncclResult_t tcpx_test(void* request, int* done, int* size)
{
	struct nccl_net_socket_request *req = request;
	struct nccl_net_socket_comm *comm = req->comm;

	int data = req->size;
	int len;

	if (req == NULL) {
		log(WARN, "failed to tcpx_test(): request is NULL");
		return ncclInternalError;
	}

	if (req->used == 0) {
		log(WARN, "failed to tcpx_test(): req->used is zero");
		return ncclInvalidUsage;
	}

	if (req->op == NCCL_SOCKET_RECV) {
		len = recv_all(comm->fd, &data, sizeof(int));
		if (len < 0) {
			log(PWARN, "failed to recv_all(): ");
			return ncclInternalError;
		}

		if (len == 0) {
			log(WARN, "failed to tcpx_test(): connection closed");
			return ncclRemoteError;
		}

	} else if (req->op == NCCL_SOCKET_SEND) {
		len = send_all(comm->fd, &data, sizeof(int));
		if (len < 0) {
			log(PWARN, "failed to send_all(): ");
			return ncclInternalError;
		}

		if (len == 0) {
			log(WARN, "failed to tcpx_test(): connection closed");
			return ncclRemoteError;
		}
	}

	req->size = data;

	if (req->op == NCCL_SOCKET_RECV)
		len = recv_all(comm->fd, req->data, req->size);
	else if (req->op == NCCL_SOCKET_SEND)
		len = send_all(comm->fd, req->data, req->size);

	if (len < 0) {
		log(PWARN, "failed to recv_all(): ");
		return ncclInternalError;
	}

	if (len == 0) {
		log(WARN, "failed to tcpx_test(): connection closed");
		return ncclRemoteError;
	}

	*done = 1;
	*size = req->size;
	req->used = 0;

	log(INFO, "tcpx_test() complete:");
	log(INFO, "\treq->comm->dev: %d", req->comm->dev);
	log(INFO, "\treq->comm->fd: %d", req->comm->fd);
	log(INFO, "\treq->op: %d", req->op);

	return ncclSuccess;
}

__hidden ncclResult_t tcpx_close_send(void* sendComm)
{
	struct nccl_net_socket_comm *comm = sendComm;

	log(INFO, "tcpx_close_send() complete: ");
	log(INFO, "\tcomm->fd: %d", comm->fd);

	close(comm->fd);
	free(comm);

	return ncclSuccess;
}

__hidden ncclResult_t tcpx_close_recv(void* recvComm)
{
	struct nccl_net_socket_comm *comm = recvComm;

	log(INFO, "tcpx_close_recv() complete: ");
	log(INFO, "\tcomm->fd: %d", comm->fd);

	close(comm->fd);
	free(comm);

	return ncclSuccess;
}

__hidden ncclResult_t tcpx_close_listen(void* listenComm)
{
	struct nccl_net_socket_listen_comm *comm = listenComm;

	log(INFO, "tcpx_close_listen() complete: ");
	log(INFO, "\tcomm->fd: %d", comm->fd);

	close(comm->fd);
	free(comm);

	return ncclSuccess;
}

__hidden ncclResult_t tcpx_irecv_consumed(void* recvComm, int n, void* request)
{
	log(WARN, "tcpx_irecv_consumed() is not implemented yet.");
	return ncclSuccess;
}

__hidden ncclResult_t tcpx_get_device_mr(void* comm, void* mhandle,
					void** dptr_mhandle)
{
	log(WARN, "tcpx_get_device_mr() is not implemented yet.");
	return ncclSuccess;
}

__hidden ncclResult_t tcpx_make_vdevice(int* d, ncclNetVDeviceProps_t* props)
{
	log(WARN, "tcpx_make_vdevice() is not implemented yet.");
	return ncclSuccess;
}

__hidden ncclResult_t tcpx_get_properties_v9(int dev,
					     ncclNetProperties_v9_t *props)
{
	/* Below are default values, if unsure don't change. */
	props->name = "Example";
	/* Fill for proper topology detection, e.g.
	 * /sys/devices/pci0000:00/0000:00:10.0/0000:0b:00.0
	 */
	props->pciPath = NULL;
	/* Only used to detect NICs with multiple PCI attachments. */
	props->guid = 0;
	/* Add NCCL_PTR_CUDA if GPU Direct RDMA is supported and regMr can
	 * take CUDA pointers.
	 */
	props->ptrSupport = NCCL_PTR_HOST;
	/* If you regMr has a fast registration cache, set to 1.
	 * If set to 0, user buffer registration may be disabled.
	 */
	props->regIsGlobal = 0;
	/* Force flush after receive. Needed if the control path and data
	 * path use a different path to the GPU
	 */
	props->forceFlush = 0;
	/* Speed in *Mbps*. 100000 means 100G */
	props->speed = 100000;
	/* Port number, used in conjunction with guid */
	props->port = 0;
	/* Custom latency (used to help tuning if latency is high.
	 * If set to 0, use default NCCL values.
	 */
	props->latency = 0;
	/* Maximum number of comm objects we can create. */
	props->maxComms = 1024*1024;
	/* Maximum number of receive operations taken by irecv(). */
	props->maxRecvs = NCCL_PLUGIN_MAX_RECVS;
	/* Coupling with NCCL network device-side code. */
	props->netDeviceType = NCCL_NET_DEVICE_HOST;
	props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
	/* Used to tell NCCL core whether this is a virtual device fusing
	 * multiple physical devices.
	 */
	props->vProps.ndevs = 1;
	props->vProps.devs[0] = dev;
	/* maximum transfer sizes the plugin can handle */
	props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
	props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;

	log(INFO, "tcpx_get_properties_v9() complete: device %d", dev);

	return ncclSuccess;
}

#define PLUGIN_NAME "tcpx"

ncclNet_v9_t ncclNetPlugin_v9 = {
	.name = PLUGIN_NAME,
	.init = tcpx_init,
	.devices = tcpx_devices,
	.getProperties = tcpx_get_properties_v9,
	.listen = tcpx_listen,
	.connect = tcpx_connect,
	.accept = tcpx_accept,
	.regMr = tcpx_reg_mr,
	.regMrDmaBuf = tcpx_reg_mr_dmabuf,
	.deregMr = tcpx_dereg_mr,
	.isend = tcpx_isend,
	.irecv = tcpx_irecv,
	.iflush = tcpx_iflush,
	.test = tcpx_test,
	.closeSend = tcpx_close_send,
	.closeRecv = tcpx_close_recv,
	.closeListen = tcpx_close_listen,
	.getDeviceMr = tcpx_get_device_mr,
	.irecvConsumed = tcpx_irecv_consumed,
	.makeVDevice   = tcpx_make_vdevice,
};

__hidden ncclResult_t tcpx_get_properties_v8(int dev,
					     ncclNetProperties_v8_t* props)
{
	/* Below are default values, if unsure don't change. */
	props->name = "Example";
	/* Fill for proper topology detection, e.g.
	 * /sys/devices/pci0000:00/0000:00:10.0/0000:0b:00.0
	 */
	props->pciPath = NULL;
	/* Only used to detect NICs with multiple PCI attachments. */
	props->guid = 0;
	/* Add NCCL_PTR_CUDA if GPU Direct RDMA is supported and regMr can
	 * take CUDA pointers.
	 */
	props->ptrSupport = NCCL_PTR_HOST;
	/* If you regMr has a fast registration cache, set to 1.
	 * If set to 0, user buffer registration may be disabled.
	 */
	props->regIsGlobal = 0;
	/* Speed in *Mbps*. 100000 means 100G */
	props->speed = 100000;
	/* Port number, used in conjunction with guid */
	props->port = 0;
	/* Custom latency (used to help tuning if latency is high.
	 * If set to 0, use default NCCL values.
	 */
	props->latency = 0;
	/* Maximum number of comm objects we can create. */
	props->maxComms = 1024*1024;
	/* Maximum number of receive operations taken by irecv(). */
	props->maxRecvs = NCCL_PLUGIN_MAX_RECVS;
	/* Coupling with NCCL network device-side code. */
	props->netDeviceType = NCCL_NET_DEVICE_HOST;
	props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;

	log(INFO, "tcpx_get_properties()_v8 complete: device %d", dev);

	return ncclSuccess;
}

__hidden ncclResult_t tcpx_isend_v8(void* sendComm, void* data, int size,
				  int tag, void* mhandle, void** request)
{
	size_t new_size = size;

	return ncclNetPlugin_v9.isend(
		sendComm, data, new_size, tag, mhandle, request
	);
}

__hidden ncclResult_t tcpx_irecv_v8(void* recvComm, int n, void** data,
				    int* sizes, int* tags, void** mhandles,
				    void** request)
{
	size_t new_sizes[n];

	for (int i = 0; i < n; i++)
		new_sizes[i] = sizes[i];

	return ncclNetPlugin_v9.irecv(
		recvComm, n, data, new_sizes, tags, mhandles, request
	);
}

ncclNet_v8_t ncclNetPlugin_v8 = {
	.name = PLUGIN_NAME,
	.init = tcpx_init,
	.devices = tcpx_devices,
	.getProperties = tcpx_get_properties_v8,
	.listen = tcpx_listen,
	.connect = tcpx_connect,
	.accept = tcpx_accept,
	.regMr = tcpx_reg_mr,
	.regMrDmaBuf = tcpx_reg_mr_dmabuf,
	.deregMr = tcpx_dereg_mr,
	.isend = tcpx_isend_v8,
	.irecv = tcpx_irecv_v8,
	.iflush = tcpx_iflush,
	.test = tcpx_test,
	.closeSend = tcpx_close_send,
	.closeRecv = tcpx_close_recv,
	.closeListen = tcpx_close_listen,
	.getDeviceMr = tcpx_get_device_mr,
	.irecvConsumed = tcpx_irecv_consumed,
};
