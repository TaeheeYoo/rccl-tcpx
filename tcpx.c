#include "tcpx.h"

int max_requests = NCCL_NET_MAX_REQUESTS;
int ncclNetIfs = 0;

struct nccl_net_socket_comm {
	int fd;
	int num_socks;
	int num_threads;
	int dev;
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
	struct ifaddrs *ifaddr, *ifa;
	int i = 0, count = 0;
	char *token;

	if (!ifs) {
		printf("NET/TCPX tcpx interfaces are not defined\n");
		return ncclInternalError;
	} else {
		if (getifaddrs(&ifaddr) == -1) {
			printf("NET/TCPX Can't get interfaces\n");
			return ncclInternalError;
		}

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
					printf("NET/TCPX Interface %d: %s\n",
					       i + 1, tcpx_devs[i].dev_name);
					i++;
					ncclNetIfs++;
					goto next;
				}
			}
next:
			count++;
			token = strtok(NULL, ",");
		}

		if (i == MAX_IFS) {
			printf("Maximum number of interfaces reached.\n");
		}
	}

	return ncclSuccess;
}

__hidden ncclResult_t pluginDevices(int* ndev)
{
	printf("[TEST]%s %u\n", __func__, __LINE__);

	/* interface index? */
	*ndev = ncclNetIfs;

	return ncclSuccess;
}

__hidden ncclResult_t pluginPciPath(int dev, char** path)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);

	return ncclInternalError;
}

__hidden ncclResult_t pluginPtrSupport(int dev, int* supportedTypes)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);

	return ncclInternalError;
}

__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_t* props)
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
	return ncclSuccess;
}

__hidden ncclResult_t tcpx_listen(int dev, void *opaque_handle,
				  void **listen_comm)
{
	struct nccl_net_socket_listen_comm *comm;
	int family, salen, err, sockfd, opt = 1;
	struct nccl_net_socket_handle* handle;
	char line[SOCKET_NAME_MAXLEN + 1];
	ncclResult_t retval;

	handle = (struct nccl_net_socket_handle *)opaque_handle;

	if (dev < 0 || dev >= ncclNetIfs) {
		printf("NET/TCPX: tcpx_listen dev=%d ncclNetIfs=%d",
		       dev, ncclNetIfs);
		retval = ncclInternalError;
		goto RETURN_ERROR;
	}

	comm = calloc(1, sizeof(struct nccl_net_socket_listen_comm));
	if (!comm) {
		printf("NET/TCPX: Failed to allocate memory\n");
		retval = ncclInternalError;
		goto RETURN_ERROR;
	}

	comm->fd = -1;
	memcpy((void *)&handle->connect_addr, &tcpx_devs[dev].addr,
	       sizeof(handle->connect_addr));

	family = handle->connect_addr.sa.sa_family;
	printf("[TEST]%s %u dev=%d family = %d \n", __func__, __LINE__, dev, family);
	salen = (family == AF_INET) ? sizeof(struct sockaddr_in) :
				      sizeof(struct sockaddr_in6);

	/* Create socket and bind it to a port */
	sockfd = socket(family, SOCK_STREAM, 0);
	if (sockfd == -1) {
		printf("NET/TCPX: Socket creation failed : %s\n",
		       strerror(errno));
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
			printf("NET/TCPX : setsockopt failed\n");
			retval = ncclSystemError;
			goto CLOSE_SOCKFD;
		}
	}

	err = bind(sockfd, &handle->connect_addr.sa, salen);
	if (err) {
		printf("NET/TCPX : bind failed\n");
		retval = ncclSystemError;
		goto CLOSE_SOCKFD;
	}

	/* Get the assigned Port */
	socklen_t size = salen;
	err = getsockname(sockfd, &handle->connect_addr.sa, &size);
	if (err) {
		printf("NET/TCPX : getsockname failed\n");
		retval = ncclSystemError;
		goto CLOSE_SOCKFD;
	}

	/* Put the socket in listen mode
	 * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
	 */
	err = listen(sockfd, 16384);
	if (err) {
		printf("NET/TCPX : listen failed\n");
		retval = ncclSystemError;
		goto CLOSE_SOCKFD;
	}
	comm->fd = sockfd;

	comm->num_socks = 1;
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

	comm = calloc(1, sizeof(struct nccl_net_socket_comm));
	if (comm == NULL) {
		retval = ncclInternalError;
		goto RETURN_ERROR;
	}

	comm->fd = socket(handle->connect_addr.sa.sa_family, SOCK_STREAM, 0);
	if (comm->fd == -1) {
		retval = ncclSystemError;
		goto FREE_COMM;
	}

	addrlen = (handle->connect_addr.sa.sa_family == AF_INET) ?
		  sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
	if (connect(comm->fd, &handle->connect_addr.sa, addrlen) == -1) {
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

	rcomm = calloc(1, sizeof(struct nccl_net_socket_comm));
	if (rcomm == NULL) {
		retval = ncclInternalError;
		goto RETURN_ERROR;
	}

	rcomm->num_socks = lcomm->num_socks;
	rcomm->num_threads = lcomm->num_threads;
	rcomm->dev = lcomm->dev;

	rcomm->fd = accept(lcomm->fd, NULL, 0);
	if (rcomm->fd == -1) {
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
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginRegMrDmaBuf(void* collComm, void* data, size_t size,
					int type, uint64_t offset, int fd,
					void** mhandle)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginDeregMr(void* collComm, void* mhandle)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginIsend(void* sendComm, void* data, size_t size,
				  int tag, void* mhandle, void** request)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data,
				  size_t* sizes, int* tags, void** mhandles,
				  void** request)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginIflush(void* recvComm, int n, void** data,
				   int* sizes, void** mhandles, void** request)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginTest(void* request, int* done, int* size)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginCloseSend(void* sendComm)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginCloseRecv(void* recvComm)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginCloseListen(void* listenComm)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginIrecvConsumed(void* recvComm, int n, void* request)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginGetDeviceMr(void* comm, void* mhandle,
					void** dptr_mhandle)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

__hidden ncclResult_t pluginMakeVDevice(int* d, ncclNetVDeviceProps_t* props)
{
	printf("[TEST]%s %u \n", __func__, __LINE__);
	return ncclInternalError;
}

#define PLUGIN_NAME "tcpx"

ncclNet_v9_t ncclNetPlugin_v9 = {
	.name = PLUGIN_NAME,
	.init = tcpx_init,
	.devices = pluginDevices,
	.getProperties = pluginGetProperties,
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

__hidden ncclResult_t pluginGetProperties_v8(int dev, ncclNetProperties_v8_t* props_v8) {
	ncclNetProperties_t props;
	ncclResult_t ret = pluginGetProperties(dev, &props);
	if (ret != ncclSuccess) return ret;
	props_v8->name = props.name;
	props_v8->pciPath = props.pciPath;
	props_v8->guid = props.guid;
	props_v8->ptrSupport = props.ptrSupport;
	props_v8->regIsGlobal = props.regIsGlobal;
	props_v8->speed = props.speed;
	props_v8->latency = props.latency;
	props_v8->port = props.port;
	props_v8->maxComms = props.maxComms;
	props_v8->maxRecvs = props.maxRecvs;
	props_v8->netDeviceType = props.netDeviceType;
	props_v8->netDeviceVersion = props.netDeviceVersion;
	return ncclSuccess;
}

__hidden ncclResult_t pluginIsend_v8(void* sendComm, void* data, int size,
				     int tag, void* mhandle, void** request)
{
	return pluginIsend(sendComm, data, (int)size, tag, mhandle, request);
}

__hidden ncclResult_t pluginIrecv_v8(void* recvComm, int n, void** data,
				     int* sizes, int* tags, void** mhandles,
				     void** request)
{
	size_t sizesOut[NCCL_PLUGIN_MAX_RECVS];
	int i;

	for (i = 0; i < n; i++)
		sizesOut[i] = sizes[i];
	return pluginIrecv(recvComm, 1, data, sizesOut, tags, mhandles, request);
}

const ncclNet_v8_t ncclNetPlugin_v8 = {
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

__hidden ncclResult_t pluginGetProperties_v7(int dev,
					     ncclNetProperties_v7_t* props_v7)
{
	ncclNetProperties_t props;
	ncclResult_t ret = pluginGetProperties(dev, &props);
	if (ret != ncclSuccess)
		return ret;

	props_v7->name = props.name;
	props_v7->pciPath = props.pciPath;
	props_v7->guid = props.guid;
	props_v7->ptrSupport = props.ptrSupport;
	props_v7->speed = props.speed;
	props_v7->latency = props.latency;
	props_v7->port = props.port;
	props_v7->maxComms = props.maxComms;
	props_v7->maxRecvs = props.maxRecvs;
	props_v7->netDeviceType = props.netDeviceType;
	props_v7->netDeviceVersion = props.netDeviceVersion;

	return ncclSuccess;
}

__hidden ncclResult_t pluginRegMr_v7(void* collComm, void* data, int size,
				     int type, void** mhandle)
{
	return pluginRegMr(collComm, data, size, type, mhandle);
}

const ncclNet_v7_t ncclNetPlugin_v7 = {
	.name = PLUGIN_NAME,
	.init = tcpx_init,
	.devices = pluginDevices,
	.getProperties = pluginGetProperties_v7,
	.listen = tcpx_listen,
	.connect = pluginConnect,
	.accept = pluginAccept,
	.regMr = pluginRegMr_v7,
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

__hidden ncclResult_t pluginGetProperties_v6(int dev,
					     ncclNetProperties_v6_t* props_v6)
{
	ncclNetProperties_t props;
	ncclResult_t ret = pluginGetProperties(dev, &props);
	if (ret != ncclSuccess)
		return ret;

	props_v6->name = props.name;
	props_v6->pciPath = props.pciPath;
	props_v6->guid = props.guid;
	props_v6->ptrSupport = props.ptrSupport;
	props_v6->speed = props.speed;
	props_v6->latency = props.latency;
	props_v6->port = props.port;
	props_v6->maxComms = props.maxComms;
	props_v6->maxRecvs = props.maxRecvs;

	return ncclSuccess;
}

__hidden ncclResult_t pluginConnect_v6(int dev, void* handle, void** sendComm)
{
	return ncclInternalError;
}

__hidden ncclResult_t pluginAccept_v6(void* listenComm, void** recvComm)
{
	return ncclInternalError;
}

const ncclNet_v6_t ncclNetPlugin_v6 = {
	.name = PLUGIN_NAME,
	.init = tcpx_init,
	.devices = pluginDevices,
	.getProperties = pluginGetProperties_v6,
	.listen = tcpx_listen,
	.connect = pluginConnect_v6,
	.accept = pluginAccept_v6,
	.regMr = pluginRegMr_v7,
	.regMrDmaBuf = pluginRegMrDmaBuf,
	.deregMr = pluginDeregMr,
	.isend = pluginIsend_v8,
	.irecv = pluginIrecv_v8,
	.iflush = pluginIflush,
	.test = pluginTest,
	.closeSend = pluginCloseSend,
	.closeRecv = pluginCloseRecv,
	.closeListen = pluginCloseListen
};

/* v5 Compat */
const ncclNet_v5_t ncclNetPlugin_v5 = {
	.name = PLUGIN_NAME,
	.init = tcpx_init,
	.devices = pluginDevices,
	.getProperties = pluginGetProperties_v6,
	.listen = tcpx_listen,
	.connect = pluginConnect_v6,
	.accept = pluginAccept_v6,
	.regMr = pluginRegMr_v7,
	.deregMr = pluginDeregMr,
	.isend = pluginIsend_v8,
	.irecv = pluginIrecv_v8,
	.iflush = pluginIflush,
	.test = pluginTest,
	.closeSend = pluginCloseSend,
	.closeRecv = pluginCloseRecv,
	.closeListen = pluginCloseListen,
};

/* v4 Compat */
static ncclResult_t pluginGetProperties_v4(int dev, ncclNetProperties_v4_t* props_v4) {
	ncclNetProperties_t props;
	ncclResult_t ret = pluginGetProperties(dev, &props);
	if (ret != ncclSuccess) return ret;
	props_v4->name = props.name;
	props_v4->pciPath = props.pciPath;
	props_v4->guid = props.guid;
	props_v4->ptrSupport = props.ptrSupport;
	props_v4->speed = props.speed;
	props_v4->port = props.port;
	props_v4->maxComms = props.maxComms;
	return ncclSuccess;
}

static ncclResult_t pluginIsend_v4(void *sendComm, void* data, int size,
				   void *mhandle, void** request)
{
	return pluginIsend_v8(sendComm, data, size, 0, mhandle, request);
}

static ncclResult_t pluginIrecv_v4(void* recvComm, void* data, int size,
				   void* mhandle, void** request)
{
	int tag = 0;

	return pluginIrecv_v8(recvComm, 1, &data, &size, &tag, &mhandle,
			      request);
}

static ncclResult_t pluginIflush_v4(void* recvComm, void* data, int size,
				    void* mhandle, void** request)
{
	return pluginIflush(recvComm, 1, &data, &size, &mhandle, request);
}

static ncclResult_t pluginConnect_v4(int dev, void* handle, void** sendComm)
{
	ncclResult_t ret;

	do {
		ncclNetDeviceHandle_v7_t* handle = NULL;
		ret = pluginConnect(dev, handle, sendComm, &handle);
	} while (ret == ncclSuccess && *sendComm == NULL);

	return ret;
}

static ncclResult_t pluginAccept_v4(void* listenComm, void** recvComm)
{
	ncclResult_t ret;

	do {
		ncclNetDeviceHandle_v7_t* handle = NULL;
		ret = pluginAccept(listenComm, recvComm, &handle);
	} while (ret == ncclSuccess && *recvComm == NULL);

	return ret;
}

const ncclNet_v4_t ncclNetPlugin_v4 = {
	.name = PLUGIN_NAME,
	.init = tcpx_init,
	.devices = pluginDevices,
	.getProperties = pluginGetProperties_v4,
	.listen = tcpx_listen,
	.connect = pluginConnect_v4,
	.accept = pluginAccept_v4,
	.regMr = pluginRegMr_v7,
	.deregMr = pluginDeregMr,
	.isend = pluginIsend_v4,
	.irecv = pluginIrecv_v4,
	.iflush = pluginIflush_v4,
	.test = pluginTest,
	.closeSend = pluginCloseSend,
	.closeRecv = pluginCloseRecv,
	.closeListen = pluginCloseListen,
};

/* v3 Compat */
static ncclResult_t pluginFlush(void* recvComm, void* data, int size,
				void* mhandle)
{
	ncclResult_t ret;
	int done = 0;
	void* req;

	ret = pluginIflush_v4(recvComm, data, size, mhandle, &req);

	while (ret == ncclSuccess && done == 0)
		ret = pluginTest(req, &done, NULL);

	return ret;
}

static ncclResult_t tcpx_init_v3(ncclDebugLogger_t logFunction)
{
	max_requests = NCCL_NET_MAX_REQUESTS_V3;

	return tcpx_init(logFunction);
}

static ncclResult_t tcpx_listen_v3(int dev, void* handle, void** listenComm)
{
	char pluginHandle[NCCL_NET_HANDLE_MAXSIZE];
	ncclResult_t ret;

	ret = tcpx_listen(dev, &pluginHandle, listenComm);

	memcpy(handle, &pluginHandle, NCCL_NET_HANDLE_MAXSIZE_V4);

	return ret;
}

static ncclResult_t pluginConnect_v3(int dev, void* handle, void** sendComm)
{
	char pluginHandle[NCCL_NET_HANDLE_MAXSIZE];

	memcpy(&pluginHandle, handle, NCCL_NET_HANDLE_MAXSIZE_V4);

	return pluginConnect_v4(dev, &pluginHandle, sendComm);
}

const ncclNet_v3_t ncclNetPlugin_v3 = {
	.name = PLUGIN_NAME,
	.init = tcpx_init_v3,
	.devices = pluginDevices,
	.getProperties = pluginGetProperties_v4,
	.listen = tcpx_listen_v3,
	.connect = pluginConnect_v3,
	.accept = pluginAccept_v4,
	.regMr = pluginRegMr_v7,
	.deregMr = pluginDeregMr,
	.isend = pluginIsend_v4,
	.irecv = pluginIrecv_v4,
	.flush = pluginFlush,
	.test = pluginTest,
	.closeSend = pluginCloseSend,
	.closeRecv = pluginCloseRecv,
	.closeListen = pluginCloseListen,
};

/* v2 Compat */
const ncclNet_v2_t ncclNetPlugin_v2 = {
	.name = PLUGIN_NAME,
	.init = tcpx_init_v3,
	.devices = pluginDevices,
	.pciPath = pluginPciPath,
	.ptrSupport = pluginPtrSupport,
	.listen = tcpx_listen,
	.connect = pluginConnect_v4,
	.accept = pluginAccept_v4,
	.regMr = pluginRegMr_v7,
	.deregMr = pluginDeregMr,
	.isend = pluginIsend_v4,
	.irecv = pluginIrecv_v4,
	.flush = pluginFlush,
	.test = pluginTest,
	.closeSend = pluginCloseSend,
	.closeRecv = pluginCloseRecv,
	.closeListen = pluginCloseListen,
};
