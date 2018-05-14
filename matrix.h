#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"

typedef struct Matrix
{
    int row, col;
    float **vals;
} Matrix;

Matrix create_matrix(int Mat_Row, int Mat_Col);
Matrix constant_matrix(int Mat_Row, int Mat_Col, float val);
Matrix random_matrix(int Mat_Row, int Mat_Col);
void print_matrix(Matrix *Mat);
void print_matrix_dim(Matrix *Mat);
void free_matrix(Matrix *Mat);
Matrix copy_Mat_wCPU(Matrix *Mat_In);
Matrix transpose_Mat_wCPU(Matrix *Mat_In);

#endif // MATRIX_H
