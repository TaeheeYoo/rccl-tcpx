#include "tcpx.h"

#include "net_v9.h"
#include "net_v8.h"

#include "Cruzer-S/logger/logger.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>

int max_requests = NCCL_NET_MAX_REQUESTS;
int ncclNetIfs = 0;
int sport = 50000;

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

	struct nccl_net_socket_request requests[MAX_REQUESTS];
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
	char dev_name[MAX_IF_NAME_SIZE];
	char* pci_path;
};

static struct tcpx_dev tcpx_devs[MAX_IFS];

/* TODO NCCL_TCPX_SUBNET */
__hidden ncclResult_t tcpx_init(ncclDebugLogger_t logFunction)
{
	char* ifs = getenv("NCCL_TCPX_IFNAMES");
	char *port = getenv("NCCL_TCPX_PORT");
	struct ifaddrs *ifaddr, *ifa;
	int i = 0, count = 0;
	char *token;

	if (!ifs) {
		fprintf(stderr, "NET/TCPX tcpx interfaces are not defined\n");
		return ncclInternalError;
	}

	if (port)
		sport = strtol(port, NULL, 10);

	if (getifaddrs(&ifaddr) == -1) {
		fprintf(stderr, "NET/TCPX Can't get interfaces\n");
		return ncclInternalError;
	}

	if (!logger_initialize()) {
		fprintf(stderr, "failed to logger_initialize(): %s",
				strerror(errno));
	}

	log(INFO, "Port Number: %d", sport);

	token = strtok(ifs, ",");
	while (token != NULL && count < MAX_IFS) {
		for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
			if (ifa->ifa_addr &&
			    !strcmp(ifa->ifa_name, token)) {
				if (ifa->ifa_addr->sa_family != AF_INET)
					continue;

				memcpy(&tcpx_devs[i].addr,
				       ifa->ifa_addr,
				       sizeof(struct sockaddr));
				strcpy(tcpx_devs[i].dev_name, token);
				log(INFO, "NET/TCPX Interface %d: %s",
					  i + 1, tcpx_devs[i].dev_name);
				i++;
				ncclNetIfs++;
				goto next;
			}
		}

	next:	count++;
		token = strtok(NULL, ",");
	}

	if (i == MAX_IFS)
		log(WARN, "Maximum number of interfaces reached.");

	return ncclSuccess;
}

__hidden ncclResult_t pluginDevices(int* ndev)
{
	log(INFO, "Set devices");

	/* interface index? */
	*ndev = ncclNetIfs;

	return ncclSuccess;
}

__hidden ncclResult_t pluginPciPath(int dev, char** path)
{
	log(INFO, "Set PCI path: %s", path);

	return ncclSuccess;
}

__hidden ncclResult_t pluginPtrSupport(int dev, int* supportedTypes)
{
	log(INFO, "Return Supported Types");

	return ncclSuccess;
}

__hidden ncclResult_t pluginGetProperties_v8(int dev, ncclNetProperties_v8_t* props)
{
	log(INFO, "Return properties");

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

	return ncclSuccess;
}

__hidden ncclResult_t tcpx_listen(int dev, void *opaque_handle,
				  void **listen_comm)
{
	struct nccl_net_socket_listen_comm *comm;
	struct sockaddr_in *addr;
	int family, salen, err, sockfd, opt = 1;
	struct nccl_net_socket_handle* handle;
	char line[SOCKET_NAME_MAXLEN + 1];
	ncclResult_t retval;

	handle = (struct nccl_net_socket_handle *)opaque_handle;

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

	comm->fd = -1;
	memcpy((void *)&handle->connect_addr, &tcpx_devs[dev].addr,
	       sizeof(handle->connect_addr));

	family = handle->connect_addr.sa.sa_family;
	log(INFO, "dev=%d family = %d", dev, family);
	salen = (family == AF_INET) ? sizeof(struct sockaddr_in) :
				      sizeof(struct sockaddr_in6);

	/* Create socket and bind it to a port */
	sockfd = socket(family, SOCK_STREAM, 0);
	if (sockfd == -1) {
		log(PWARN, "Failed to socket(): ");
		retval = ncclSystemError;
		goto FREE_COMM;
	}

	if (socket_to_port(&handle->connect_addr.sa)) {
		// Port is forced by env. Make sure we get the port.
#if defined(SO_REUSEPORT)
		err = setsockopt(sockfd, SOL_SOCKET,
				 SO_REUSEADDR | SO_REUSEPORT,
				 &opt, sizeof(opt));
#else
		err = setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt,
				 sizeof(opt));
#endif
		if (err) {
			log(PWARN, "Failed to setsockopt: ");
			retval = ncclSystemError;
			goto CLOSE_SOCKFD;
		}
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
	comm->num_threads = 1;
	handle->num_socks = comm->num_socks;
	handle->num_threads = comm->num_threads;
	comm->dev = dev;
	*listen_comm = comm;

	return ncclSuccess;

CLOSE_SOCKFD:	close(sockfd);
FREE_COMM:	free(comm);
RETURN_ERROR:	return retval;
}

__hidden ncclResult_t pluginConnect(int dev, void* opaqueHandle,
				    void** sendComm,
				    ncclNetDeviceHandle_t** sendDevComm)
{
	struct nccl_net_socket_handle *handle = opaqueHandle;
	struct nccl_net_socket_comm *comm;
	ncclResult_t retval;
	size_t addrlen;

	log(INFO, "Try to connect");
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

	return ncclSuccess;

CLOSE_COMM_FD:	close(comm->fd);
FREE_COMM:	free(comm);
RETURN_ERROR:	return retval;
}

__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm,
				   ncclNetDeviceHandle_t** recvDevComm)
{
	struct nccl_net_socket_listen_comm *lcomm = listenComm;
	struct nccl_net_socket_comm *rcomm;
	ncclResult_t retval;

	log(INFO, "Try to accept");

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

	return ncclSuccess;

FREE_RCOMM:	free(rcomm);
RETURN_ERROR:	return retval;
}

__hidden ncclResult_t pluginRegMr(void* collComm, void* data, size_t size,
				  int type, void** mhandle)
{
	log(INFO, "pluginRegMr");
	log(INFO, "\tcollComm: %p", collComm);
	log(INFO, "\tdata: %p", data); 
	log(INFO, "\tsize: %zu", size);
	log(INFO, "\ttype: %d", type);
 
	return ncclSuccess;
}

__hidden ncclResult_t pluginRegMrDmaBuf(void* collComm, void* data, size_t size,
					int type, uint64_t offset, int fd,
					void** mhandle)
{
	log(INFO, "pluginRegMrDmaBuf");
	return ncclSuccess;
}

__hidden ncclResult_t pluginDeregMr(void* collComm, void* mhandle)
{
	log(INFO, "pluginDeregMr");
	return ncclSuccess;
}

__hidden ncclResult_t pluginIsend(void* sendComm, void* data, size_t size,
				  int tag, void* mhandle, void** request)
{
	struct nccl_net_socket_comm *comm = sendComm;
	struct nccl_net_socket_request *req;

	log(INFO, "Sending data");
	log(INFO, "data: %p\tsize: %zu", data, size);

	for (int i = 0; i < MAX_REQUESTS; i++) {
		req = comm->requests + i;
		if (req->used != 0)
			continue;

		req->op = NCCL_SOCKET_SEND;
		req->data = data;
		req->size = size;
		req->used = 1;
		req->comm = comm;

		*request = req;

		return ncclSuccess;
	}

	log(WARN, "Maximum request queue!");

	return ncclInternalError;
}

__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data,
				  size_t* sizes, int* tags, void** mhandles,
				  void** request)
{
	struct nccl_net_socket_comm *comm = recvComm;
	struct nccl_net_socket_request *req;

	log(INFO, "Receive data");
	log(INFO, "data: %p\tsize: %zu", data[0], sizes[0]);

	for (int i = 0; i < MAX_REQUESTS; i++) {
		req = comm->requests + i;
		if (req->used != 0)
			continue;

		req->op = NCCL_SOCKET_RECV;
		req->data = data[0];
		req->size = sizes[0];
		req->used = 1;
		req->comm = comm;

		*request = req;

		return ncclSuccess;
	}

	log(WARN, "Maximum request queue!");

	return ncclInternalError;
}

__hidden ncclResult_t pluginIflush(void* recvComm, int n, void** data,
				   int* sizes, void** mhandles, void** request)
{
	log(INFO, "Flush data");
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

__hidden ncclResult_t pluginTest(void* request, int* done, int* size)
{
	struct nccl_net_socket_request *req = request;
	struct nccl_net_socket_comm *comm = req->comm;

	int data = req->size;
	int len;

	log(INFO, "Plugin Test");

	if (req == NULL) {
		log(WARN, "failed to pluginTest(): request is NULL");
		return ncclInternalError;
	}

	if (req->used == 0) {
		log(WARN, "failed to pluginTest(): used is zero");
		return ncclInvalidUsage;
	}

	if (req->op == NCCL_SOCKET_RECV) {
		len = recv_all(comm->fd, &data, sizeof(int));
		if (len < 0) {
			log(PWARN, "failed to recv_all(): ");
			return ncclInternalError;
		}

		if (len == 0) {
			log(WARN, "get close from remote()");
			return ncclRemoteError;
		}

		if (data > req->size)
			return ncclInvalidUsage;
	} else if (req->op == NCCL_SOCKET_SEND) {
		len = send_all(comm->fd, &data, sizeof(int));
		if (len < 0) {
			log(PWARN, "failed to recv_all(): ");
			return ncclInternalError;
		}

		if (len == 0) {
			log(WARN, "get close from remote()");
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
		log(WARN, "get close from remote()");
		return ncclRemoteError;
	}

	*done = 1;
	*size = req->size;
	req->used = 0;

	return ncclSuccess;
}

__hidden ncclResult_t pluginCloseSend(void* sendComm)
{
	struct nccl_net_socket_comm *comm = sendComm;

	log(INFO, "Close send");

	close(comm->fd);
	free(comm);

	return ncclSuccess;
}

__hidden ncclResult_t pluginCloseRecv(void* recvComm)
{
	struct nccl_net_socket_comm *comm = recvComm;

	log(INFO, "Close receive");

	close(comm->fd);
	free(comm);

	return ncclSuccess;
}

__hidden ncclResult_t pluginCloseListen(void* listenComm)
{
	struct nccl_net_socket_comm *comm = listenComm;

	log(INFO, "Close listen");

	close(comm->fd);
	free(comm);

	return ncclSuccess;
}

__hidden ncclResult_t pluginIrecvConsumed(void* recvComm, int n, void* request)
{
	log(INFO, "Receive consumed");
	return ncclSuccess;
}

__hidden ncclResult_t pluginGetDeviceMr(void* comm, void* mhandle,
					void** dptr_mhandle)
{
	log(INFO, "GetDeviceMr");
	return ncclSuccess;
}

__hidden ncclResult_t pluginMakeVDevice(int* d, ncclNetVDeviceProps_t* props)
{
	log(INFO, "MakeVDevice");
	return ncclSuccess;
}

__hidden ncclResult_t pluginGetProperties_v9(int dev, ncclNetProperties_v9_t *props)
{
	log(INFO, "Return properties");

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
	props->ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
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

	return ncclSuccess;
}

#define PLUGIN_NAME "tcpx"

ncclNet_v9_t ncclNetPlugin_v9 = {
	.name = PLUGIN_NAME,
	.init = tcpx_init,
	.devices = pluginDevices,
	.getProperties = pluginGetProperties_v9,
	.listen = tcpx_listen,
	.connect = pluginConnect,
	.accept = pluginAccept,
	.regMr = pluginRegMr,
	.regMrDmaBuf = pluginRegMrDmaBuf,
	.deregMr = pluginDeregMr,
	.isend = pluginIsend,
	.irecv = pluginIrecv,
	.iflush = pluginIflush,
	.test = pluginTest,
	.closeSend = pluginCloseSend,
	.closeRecv = pluginCloseRecv,
	.closeListen = pluginCloseListen,
	.getDeviceMr = pluginGetDeviceMr,
	.irecvConsumed = pluginIrecvConsumed,
	.makeVDevice   = pluginMakeVDevice,
};

__hidden ncclResult_t pluginIsend_v8(void* sendComm, void* data, int size,
				  int tag, void* mhandle, void** request)
{
	size_t new_size = size;

	return ncclNetPlugin_v9.isend(
		sendComm, data, new_size, tag, mhandle, request
	);
}

__hidden ncclResult_t pluginIrecv_v8(void* recvComm, int n, void** data,
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
	.devices = pluginDevices,
	.getProperties = pluginGetProperties_v8,
	.listen = tcpx_listen,
	.connect = pluginConnect,
	.accept = pluginAccept,
	.regMr = pluginRegMr,
	.regMrDmaBuf = pluginRegMrDmaBuf,
	.deregMr = pluginDeregMr,
	.isend = pluginIsend_v8,
	.irecv = pluginIrecv_v8,
	.iflush = pluginIflush,
	.test = pluginTest,
	.closeSend = pluginCloseSend,
	.closeRecv = pluginCloseRecv,
	.closeListen = pluginCloseListen,
	.getDeviceMr = pluginGetDeviceMr,
	.irecvConsumed = pluginIrecvConsumed,
};
