#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

Vector Fully_Connected_wCPU(Vector *Vec_In, Matrix *Mat_Weights, Vector *Vec_Bias);

#endif // FULLY_CONNECTED_H
