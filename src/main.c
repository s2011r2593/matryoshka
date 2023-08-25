#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "nccl.h"

#include "config.h"
#include "types.h"
#include "matryoshka_util.h"
#include "helpers.h"

int main(int argc, char* argv[]) {
    printf("setting up\n");
    matNet net;
    GenerateNetwork(&net);
    linkSorter sorter;
    GenerateSorter(&sorter);
    int* link_ids = (int*) Allocate((net.n_dev * (net.n_dev - 1)) * sizeof(int) / 2);

    printf("sending\n");
    // Check Every Device Pairing

    P2PCall(&net, 0, 1, BUFFER_SIZE, WARM_UP_ITERS);
    for (int i = 0; i < TESTING_ITERS; i++) {
        float individual;
        ProfileP2P(&net, 0, 1, BUFFER_SIZE, 1);
        cudaEventElapsedTime(&individual, net.timers[0][START], net.timers[0][STOP]);
        float aggregate = individual;
        cudaEventElapsedTime(&individual, net.timers[0][START], net.timers[0][STOP]);
        aggregate += individual;

        sorter.links[sorter.n_links].edf[i] = aggregate;
        sorter.links[sorter.n_links].cost += aggregate;
    }

    int link_idx = 0;
    for (int from = 0; from < net.n_dev - 1; from++) {
        for (int to = from + 1; to < net.n_dev; to++) {
            sorter.links[sorter.n_links].cost = 0;
            P2PCall(&net, from, to, BUFFER_SIZE, WARM_UP_ITERS);
            for (int i = 0; i < TESTING_ITERS; i++) {
                float individual;
                ProfileP2P(&net, from, to, BUFFER_SIZE, 1);
                cudaEventElapsedTime(&individual, net.timers[from][START], net.timers[from][STOP]);
                float aggregate = individual;
                cudaEventElapsedTime(&individual, net.timers[to][START], net.timers[to][STOP]);
                aggregate += individual;

                sorter.links[sorter.n_links].edf[i] = aggregate;
                sorter.links[sorter.n_links].cost += aggregate;
            }
            Sort(sorter.links[sorter.n_links].edf, TESTING_ITERS);
            if (sorter.n_links > 0) {
                char is_new = 1;
                for (int i = 0; i < sorter.n_links; i++) {
                    float ks_stat = GetKS(
                        sorter.links[i].edf,
                        sorter.links[sorter.n_links].edf,
                        TESTING_ITERS,
                        TESTING_ITERS
                    );
                    if (ks_stat < ACCEPTANCE_REGION) {
                        link_ids[link_idx] = i;
                        is_new = 0;
                        break;
                    }
                }
                if (is_new) {
                    link_ids[link_idx] = sorter.n_links;
                    sorter.n_links++;
                }
            } else {
                link_ids[link_idx] = 0;
                sorter.n_links++;
            }

            link_idx++;
        }
    }

    link_idx = 0;
    printf("\n    ");
    for (int i = 0; i < net.n_dev; i++) {
        printf("%d ", i);
    }
    printf("\n\n");
    for (int i = 0; i < net.n_dev; i++) {
        printf("%d   ", i);
        for (int j = i; j >= 0; j--) {
            printf("· ");
        }
        for (int j = i + 1; j < net.n_dev; j++) {
            printf("%d ", link_ids[link_idx]);
            link_idx++;
        }
        printf("\n");
    }
    printf("\n");

    printf("cleaning up\n");
    DestroyNetwork(&net);
    DestroySorter(&sorter);
    free(link_ids);

    return 0;
}
