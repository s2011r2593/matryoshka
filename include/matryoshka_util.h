#pragma once
#include <stdlib.h>

#include "nccl.h"

typedef struct matryoshka_network_t {
    ncclComm_t* comms;
    cudaStream_t* streams;
    char** buffers;
    char** host_buffers;
    cudaEvent_t** timers;
    int n_dev;
} m_net;

void* Allocate(size_t n_bytes);
void GenerateNetwork(m_net* net_ptr);
void DestroyNetwork(m_net* net_ptr);
void P2PCall(m_net* net_ptr, int from, int to, int msg_size, int n_iter);
void ProfileP2P(m_net* net_ptr, int from, int to, int msg_size, int n_iter);
