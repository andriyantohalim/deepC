#ifndef BLAS_H
#define BLAS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"


Vector scale_Vec_wCPU(Vector *Vec_In, float scaling_factor);
Matrix scale_Mat_wCPU(Matrix *Mat_In, float scaling_factor);
Tensor scale_Tsr_wCPU(Tensor *Tsr_In, float scaling_factor);
Vector power_Vec_wCPU(Vector *Vec_In, float power);
Matrix power_Mat_wCPU(Matrix *Mat_In, float power);
Tensor power_Tsr_wCPU(Tensor *Tsr_In, float power);
Vector addition_Vec_wCPU(Vector *Vec_A, Vector *Vec_B);
Matrix addition_Mat_wCPU(Matrix *Mat_A, Matrix *Mat_B);
Tensor addition_Tsr_wCPU(Tensor *Tsr_A, Tensor *Tsr_B);
float dotproduct_Vec_wCPU(Vector *Vec_A, Vector *Vec_B);
float dotproduct_Mat_wCPU(Matrix *Mat_A, Matrix *Mat_B);
float dotproduct_Tsr_wCPU(Tensor *Tsr_A, Tensor *Tsr_B);
Matrix multiplication_Vec_wCPU(Vector *Vec_A, Vector *Vec_B);
Matrix multiplication_Mat_wCPU(Matrix *Mat_A, Matrix *Mat_B);



#endif // BLAS_H
