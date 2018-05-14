#ifndef POOLING_H
#define POOLING_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

#include "padding.h"

Matrix maxpooling_Mat_wCPU(Matrix *Mat_In, int filter_height, int filter_width, int stride);
Tensor maxpooling_Tsr_wCPU(Tensor *Tsr_In, int filter_height, int filter_width, int stride);


#endif // POOLING_H
