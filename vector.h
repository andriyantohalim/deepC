#ifndef VECTOR_H
#define VECTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

typedef struct Vector
{
	int len;
	float *vals;
} Vector;

Vector create_vector(int Vec_Len);
Vector constant_vector(int Vec_Len, float val);
Vector random_vector(int Vec_Len);
void print_vector(Vector *Vec);
void print_vector_dim(Vector *Vec);
void free_vector(Vector *Vec);
Vector copy_Vec_wCPU(Vector *Vec_In);






#endif // VECTOR_H
