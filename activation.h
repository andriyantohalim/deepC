#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

Vector ReLU_Vec_wCPU(Vector *Vec_In);
Matrix ReLU_Mat_wCPU(Matrix *Mat_In);
Tensor ReLU_Tsr_wCPU(Tensor *Tsr_In);
Vector Leaky_ReLU_Vec_wCPU(Vector *Vec_In);
Matrix Leaky_ReLU_Mat_wCPU(Matrix *Mat_In);
Tensor Leaky_ReLU_Tsr_wCPU(Tensor *Tsr_In);
Vector softmax_Vec_wCPU(Vector *Vec_In);
Matrix softmax_Mat_wCPU(Matrix *Mat_In);
Tensor softmax_Tsr_wCPU(Tensor *Tsr_In);
Vector tanh_Vec_wCPU(Vector *Vec_In);
Matrix tanh_Mat_wCPU(Matrix *Mat_In);
Tensor tanh_Tsr_wCPU(Tensor *Tsr_In);
Vector sigmoid_Vec_wCPU(Vector *Vec_In);
Matrix sigmoid_Mat_wCPU(Matrix *Mat_In);
Tensor sigmoid_Tsr_wCPU(Tensor *Tsr_In);


#endif // ACTIVATION_H
