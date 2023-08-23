#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "nccl.h"

#include "config.h"
#include "matryoshka_util.h"

int main(int argc, char* argv[]) {
    printf("setting up\n");
    m_net net;
    GenerateNetwork(&net);
    int from, to;

    printf("sending\n");
    from = 0;
    to = 1;
    // for (int i = (1 << 20); i <= BUFFER_SIZE; i <<= 1) {
    //     P2PCall(&net, from, to, i, WARM_UP_ITERS);
    //     ProfileP2P(&net, from, to, i);
        
    //     float aggregate, individual;
    //     aggregate = 0;

    //     cudaEventElapsedTime(&individual, net.timers[from][START], net.timers[from][STOP]);
    //     aggregate += individual;
    //     cudaEventElapsedTime(&individual, net.timers[to][START], net.timers[to][STOP]);
    //     aggregate += individual;

    //     aggregate /= (float) TIMING_ITERS * 2.0;

    //     printf("%d =%d=> %d, %f\n", from, i, to, aggregate);
    // }

    float aggregate, individual;

    aggregate = 0;
    P2PCall(&net, from, to, 8388608, WARM_UP_ITERS);
    for (int i = 0; i < TIMING_ITERS; i++) {
        ProfileP2P(&net, from, to, 8388608, 1);
        cudaEventElapsedTime(&individual, net.timers[from][START], net.timers[from][STOP]);
        aggregate += individual;
        cudaEventElapsedTime(&individual, net.timers[to][START], net.timers[to][STOP]);
        aggregate += individual;
    }
    aggregate /= (float) TIMING_ITERS * 2.0;
    printf("NVLink: %f\n", aggregate);

    aggregate = 0;
    to = 2;
    P2PCall(&net, from, to, 8388608, WARM_UP_ITERS);
    for (int i = 0; i < TIMING_ITERS; i++) {
        ProfileP2P(&net, from, to, 8388608, 1);
        cudaEventElapsedTime(&individual, net.timers[from][START], net.timers[from][STOP]);
        aggregate += individual;
        cudaEventElapsedTime(&individual, net.timers[to][START], net.timers[to][STOP]);
        aggregate += individual;
    }
    aggregate /= (float) TIMING_ITERS * 2.0;
    printf("Non-NVLink: %f\n", aggregate);

    aggregate = 0;
    from = 3;
    P2PCall(&net, from, to, 8388608, WARM_UP_ITERS);
    for (int i = 0; i < TIMING_ITERS; i++) {
        ProfileP2P(&net, from, to, 8388608, 1);
        cudaEventElapsedTime(&individual, net.timers[from][START], net.timers[from][STOP]);
        aggregate += individual;
        cudaEventElapsedTime(&individual, net.timers[to][START], net.timers[to][STOP]);
        aggregate += individual;
    }
    aggregate /= (float) TIMING_ITERS * 2.0;
    printf("NVLink: %f\n", aggregate);

    // printf("cleaning up\n");
    DestroyNetwork(&net);

    return 0;
}
