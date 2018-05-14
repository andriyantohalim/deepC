#ifndef DATA_CONVERSION_H
#define DATA_CONVERSION_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

Vector Mat2Vec_wCPU(Matrix *Mat_In);
Vector Tsr2Vec_wCPU(Tensor *Tsr_In);
Matrix Vec2Mat_wCPU(Vector *Vec_In, int Mat_Row, int Mat_Col);
Tensor Vec2Tsr_wCPU(Vector *Vec_In, int Tsr_Row, int Tsr_Col, int Tsr_Depth);

#endif // DATA_CONVERSION_H
