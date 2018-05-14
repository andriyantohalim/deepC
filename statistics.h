#ifndef STATISTICS_H
#define STATISTICS_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"


float mean_Vec_wCPU(Vector *Vec_In);
float mean_Mat_wCPU(Matrix *Mat_In);
float mean_Tsr_wCPU(Tensor *Tsr_In);
float variance_Vec_wCPU(Vector *Vec_In);
float variance_Mat_wCPU(Matrix *Mat_In);
float variance_Tsr_wCPU(Tensor *Tsr_In);
float std_dev_Vec_wCPU(Vector *Vec_In);
float std_dev_Mat_wCPU(Matrix *Mat_In);
float std_dev_Tsr_wCPU(Tensor *Tsr_In);
Vector normalization_Vec_wCPU(Vector *Vec_In);
Matrix normalization_Mat_wCPU(Matrix *Mat_In);
Tensor normalization_Tsr_wCPU(Tensor *Tsr_In);


#endif // STATISTICS_H
