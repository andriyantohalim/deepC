#ifndef PADDING_H
#define PADDING_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"


Vector padding_asymmetric_Vec_wCPU(Vector *Vec_In, int padsize);
Matrix padding_2d_asymmetric_Mat_wCPU(Matrix *Mat_In, int row_padsize, int col_padsize);
Tensor padding_2d_asymmetric_Tsr_wCPU(Tensor *Tsr_In, int row_padsize, int col_padsize);
void vpadding_asymmetric_Vec_wCPU(Vector *Vec_In, int padsize);
void vpadding_2d_asymmetric_Mat_wCPU(Matrix *Mat_In, int row_padsize, int col_padsize);
void vpadding_2d_asymmetric_Tsr_wCPU(Tensor *Tsr_In, int row_padsize, int col_padsize);
Vector padding_2d_Vec_wCPU(Vector *Vec_In, int padsize);
Matrix padding_2d_Mat_wCPU(Matrix *Mat_In, int padsize);
Tensor padding_2d_Tsr_wCPU(Tensor *Tsr_In, int padsize);
void vpadding_2d_Mat_wCPU(Matrix *Mat_In, int padsize);
void vpadding_2d_Tsr_wCPU(Tensor *Tsr_In, int padsize);

#endif // PADDING_H
