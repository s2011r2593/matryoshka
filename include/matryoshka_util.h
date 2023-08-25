#pragma once
#include <stdlib.h>

#include "nccl.h"

#include "types.h"

void GenerateNetwork(matNet* net_ptr);
void DestroyNetwork(matNet* net_ptr);
void GenerateSorter(linkSorter* s_ptr);
void DestroySorter(linkSorter* s_ptr);
void P2PCall(matNet* net_ptr, int from, int to, int msg_size, int n_iter);
void ProfileP2P(matNet* net_ptr, int from, int to, int msg_size, int n_iter);
