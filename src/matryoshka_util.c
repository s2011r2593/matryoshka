#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "config.h"

#include "matryoshka_util.h"

void* Allocate(size_t n_bytes) {
    void* ptr = malloc(n_bytes);
    if (ptr == 0) {
        printf("Failed to allocate memory\n");
        exit(-1);
    }
    return ptr;
}

void GenerateNetwork(m_net* net_ptr) {
    cudaGetDeviceCount(&(net_ptr->n_dev));
    int* dev_ids = (int*) Allocate(net_ptr->n_dev * sizeof(int));
    net_ptr->comms = (ncclComm_t*) Allocate(net_ptr->n_dev * sizeof(ncclComm_t));
    for (int i = 0; i < net_ptr->n_dev; i++) {
        dev_ids[i] = i;
    }
    ncclCommInitAll(net_ptr->comms, net_ptr->n_dev, dev_ids);
    free(dev_ids);

    net_ptr->buffers = (char**) Allocate(net_ptr->n_dev * sizeof(char*));
    net_ptr->timers = (cudaEvent_t**) Allocate(net_ptr->n_dev * sizeof(cudaEvent_t*));
    net_ptr->streams = (cudaStream_t*) Allocate(net_ptr->n_dev * sizeof(cudaStream_t));
    #if CHECK_CORRECTNESS
    net_ptr->host_buffers = (char**) Allocate(net_ptr->n_dev * sizeof(char*));
    #endif
    for (int i = 0; i < net_ptr->n_dev; i++) {
        cudaSetDevice(i);
        cudaMalloc((void**) net_ptr->buffers + i, BUFFER_SIZE);
        cudaStreamCreate(net_ptr->streams + i);
        net_ptr->timers[i] = (cudaEvent_t*) Allocate(2 * sizeof(cudaEvent_t));
        cudaEventCreate(net_ptr->timers[i] + START);
        cudaEventCreate(net_ptr->timers[i] + STOP);
        #if CHECK_CORRECTNESS
        cudaMemset(net_ptr->buffers[i], i, BUFFER_SIZE);
        net_ptr->host_buffers[i] = (char*) Allocate(BUFFER_SIZE);
        #endif
    }

    return;
}

void DestroyNetwork(m_net* net_ptr) {
    for (int i = 0; i < net_ptr->n_dev; i++) {
        cudaSetDevice(i);
        cudaFree(net_ptr->buffers[i]);
        cudaStreamDestroy(net_ptr->streams[i]);
        cudaEventDestroy(net_ptr->timers[i][START]);
        cudaEventDestroy(net_ptr->timers[i][STOP]);
        free(net_ptr->timers[i]);
        #if CHECK_CORRECTNESS
        free(net_ptr->host_buffers[i]);
        #endif
    }
    free(net_ptr->buffers);
    free(net_ptr->timers);
    free(net_ptr->streams);
    #if CHECK_CORRECTNESS
    free(net_ptr->host_buffers);
    #endif

    for (int i = 0; i < net_ptr->n_dev; i++) {
        ncclCommDestroy(net_ptr->comms[i]);
    }
    free(net_ptr->comms);
}

void P2PCall(m_net* n, int from, int to, int msg_size, int n_iter) {
    for (int i = n_iter; i > 0; i--) {
        ncclGroupStart();
        ncclSend(n->buffers[from], msg_size, ncclUint8, to, n->comms[from], n->streams[from]);
        ncclRecv(n->buffers[to], msg_size, ncclUint8, from, n->comms[to], n->streams[to]);
        ncclGroupEnd();
    }
}
void ProfileP2P(m_net* n, int from, int to, int msg_size, int n_iter) {
    cudaEventRecord(n->timers[from][START], n->streams[from]);
    cudaEventRecord(n->timers[to][START], n->streams[to]);

    P2PCall(n, from, to, msg_size, n_iter);

    cudaEventRecord(n->timers[from][STOP], n->streams[from]);
    cudaEventRecord(n->timers[to][STOP], n->streams[to]);
    cudaEventSynchronize(n->timers[from][STOP]);
    cudaEventSynchronize(n->timers[to][STOP]);
    cudaStreamSynchronize(n->streams[from]);
    cudaStreamSynchronize(n->streams[to]);

    #if CHECK_CORRECTNESS
    cudaSetDevice(from);
    cudaMemcpy(n->host_buffers[from], n->buffers[from], msg_size, cudaMemcpyDeviceToHost);
    cudaMemset(n->buffers[from], from, BUFFER_SIZE);
    cudaSetDevice(to);
    cudaMemcpy(n->host_buffers[to], n->buffers[to], msg_size, cudaMemcpyDeviceToHost);
    cudaMemset(n->buffers[to], to, BUFFER_SIZE);

    for (int i = 0; i < msg_size; i++) {
        if (n->host_buffers[to][i] != n->host_buffers[from][i]) {
            printf("p2p failed\n");
            exit(-69);
        }
    }
    #endif
}
