#pragma once

#include "types.h"

void* Allocate(size_t n_bytes);
void Sort(float* data, int length);
float GetKS(float* M, float* N, int m, int n);
