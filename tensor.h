#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

typedef struct Tensor
{
    int depth, row, col;
    float ***vals;
} Tensor;

Tensor create_tensor(int Tsr_Row, int Tsr_Col, int Tsr_Depth);
Tensor constant_tensor(int Tsr_Row, int Tsr_Col, int Tsr_Depth, float val);
Tensor random_tensor(int Tsr_Row, int Tsr_Col, int Tsr_Depth);
void print_tensor(Tensor *Tsr);
void print_tensor_dim(Tensor *Tsr);
void free_tensor(Tensor *Tsr);
Tensor copy_Tsr_wCPU(Tensor *Tsr_In);




#endif // TENSOR_H
