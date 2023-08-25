#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "helpers.h"
#include "types.h"

void* Allocate(size_t n_bytes) {
    void* ptr = malloc(n_bytes);
    if (ptr == 0) {
        printf("Failed to allocate memory\n");
        exit(-1);
    }
    return ptr;
}

// PARENT: (i - 1) >> 1
// LEFT:   2 * i + 1
// RIGHT:  2 * i + 2
typedef struct binheap_t {
    float* data;
    int idx;
} binheap;
void Swap(binheap* bhp, int i, int j) {
    float tmp = bhp->data[i];
    bhp->data[i] = bhp->data[j];
    bhp->data[j] = tmp;
    return;
}
void HeapInsert(binheap* bhp, float point) {
    int idx = bhp->idx;
    bhp->data[idx] = point;
    bhp->idx++;

    int parent;
    while (1) {
        parent = (idx - 1) >> 1;
        if (idx == 0 || bhp->data[parent] < bhp->data[idx]) {
            break;
        }
        Swap(bhp, parent, idx);
        idx = parent;
    }

    return;
}
float HeapPop(binheap* bhp) {
    float ret = bhp->data[0];
    bhp->idx--;
    bhp->data[0] = bhp->data[bhp->idx];
    int me, left, right, min;
    me = 0;
    while (1) {
        left = (2 * me) + 1;
        right = (2 * me) + 2;
        min = me;
        if (left < bhp->idx && bhp->data[left] < bhp->data[min]) {
            min = left;
        }
        if (right < bhp->idx && bhp->data[right] < bhp->data[min]) {
            min = right;
        }
        if (min == me){
            break;
        }
        Swap(bhp, me, min);
        me = min;
    }
    return ret;
}

void Sort(float* data, int len) {
    binheap bhp;
    bhp.data = (float*) Allocate(len * sizeof(float));
    bhp.idx = 0;

    for (int i = 0; i < len; i++) {
        HeapInsert(&bhp, data[i]);
    }
    for (int i = 0; i < len; i++) {
        data[i] = HeapPop(&bhp);
    }

    free(bhp.data);
}

float GetKS(float* M, float* N, int m, int n) {
    binheap bhp;
    bhp.data = (float*) Allocate((m + n) * sizeof(float));
    bhp.idx = 0;
    for (int i = 0; i < m; i++) {
        HeapInsert(&bhp, M[i]);
    }
    for (int i = 0; i < n; i++) {
        HeapInsert(&bhp, N[i]);
    }

    float supremum = 0.0;
    float comp;
    int mi = 0;
    int ni = 0;
    float mv = 0;
    float nv = 0;
    for (int i = 0; i < m + n; i++) {
        comp = HeapPop(&bhp);
        if (M[mi] == comp) {
            mi++;
            mv = (float) mi / (float) m;
        }
        if (N[ni] == comp) {
            ni++;
            nv = (float) ni / (float) n;
        }

        if (fabsf(mv - nv) > supremum) {
            supremum = fabsf(mv - nv);
        }
    }

    free(bhp.data);

    return supremum;
}
