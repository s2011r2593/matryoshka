#pragma once
#include <stdlib.h>

#include "nccl.h"

#include "config.h"

typedef struct matryoshkaNetwork_t {
    ncclComm_t* comms;
    cudaStream_t* streams;
    char** buffers;
    char** host_buffers;
    cudaEvent_t** timers;
    int n_dev;
} matNet;

typedef struct linkType_t {
    float* edf;
    float cost;
} linkType;

typedef struct linkSorter_t {
    linkType links[NUM_LINK_TYPES];
    int n_links;
} linkSorter;
