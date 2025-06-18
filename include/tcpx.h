#ifndef THIRD_PARTY_GPUS_RCCL_TCPX_PLUGIN_UTIL_H_
#define THIRD_PARTY_GPUS_RCCL_TCPX_PLUGIN_UTIL_H_

#include "net.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <net/if.h>

#define __hidden __attribute__ ((visibility("hidden")))
#define NCCL_PLUGIN_MAX_RECVS   1
#define MAX_IF_NAME_SIZE        16
#define MAX_IFS                 16
/* connection retry sleep interval in usec */
#define SLEEP_INT               1000
/* connection refused retry times before reporting a timeout (20 sec) */
#define RETRY_REFUSED_TIMES     2e4
/* connection timed out retry times (each one can take 20s) */
#define RETRY_TIMEDOUT_TIMES    3
#define SOCKET_NAME_MAXLEN (NI_MAXHOST + NI_MAXSERV)
#define MAX_REQUESTS NCCL_NET_MAX_REQUESTS

static inline uint16_t socket_to_port(struct sockaddr *saddr) {
	return ntohs(saddr->sa_family == AF_INET ?
		     ((struct sockaddr_in*)saddr)->sin_port :
		     ((struct sockaddr_in6*)saddr)->sin6_port);
}

#endif  // THIRD_PARTY_GPUS_RCCL_TCPX_PLUGIN_UTIL_H_
