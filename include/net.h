/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#define NCCL_NET_HANDLE_MAXSIZE 128
#define NCCL_MAX_NET_SIZE_BYTES (1*1024*1024*1024*1024L) //1TB
#define NCCL_NET_OPTIONAL_RECV_COMPLETION 0x1

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

// Maximum number of requests per comm object
#define NCCL_NET_MAX_REQUESTS 32

#endif // end include guard
