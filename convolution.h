#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

#include "padding.h"

Matrix convolution_2d_Mat_wCPU(Matrix *Mat_In, Matrix *Mat_kernel, int stride);
Matrix convolution_2d_with_pad_Mat_wCPU(Matrix *Mat_In, int padsize, Matrix *Mat_kernel, int stride);

Tensor convolution_2d_Tsr_wCPU(Tensor *Tsr_In, Tensor *Tsr_kernel, int stride, int filter_size);
Tensor convolution_2d_with_pad_Tsr_wCPU(Tensor *Tsr_In, int padsize, Tensor *Tsr_kernel, int stride, int filter_size);

#endif // CONVOLUTION_H
