#include "matrix.h"

Matrix create_matrix(int Mat_Row, int Mat_Col)
{
    Matrix M;
    
    M.row = Mat_Row;
    M.col = Mat_Col;
    
    M.vals = calloc(M.row, sizeof(float *));
    
    for(int j = 0; j < M.row; j++)
    {
        M.vals[j] = calloc(M.col, sizeof(float));
    }
    
    return M;
}

Matrix constant_matrix(int Mat_Row, int Mat_Col, float val)
{
    Matrix tempMat = create_matrix(Mat_Row, Mat_Col);
    
    for (int i = 0; i < Mat_Row; i++)
    {
        for (int j = 0; j < Mat_Col; j++)
        {
            *(tempMat.vals[i] + j) = val;
        }
    }
    
    return tempMat;
}

Matrix random_matrix(int Mat_Row, int Mat_Col)
{
    Matrix tempMat = create_matrix(Mat_Row, Mat_Col);
    
    for (int i = 0; i < Mat_Row; i++)
    {
        for (int j = 0; j < Mat_Col; j++)
        {
            *(tempMat.vals[i] + j) = 2*i+j+1.3;
        }
    }
    
    return tempMat;
}

void print_matrix(Matrix *Mat)
{       
    printf("Dim: %d x %d\n", Mat->row, Mat->col);
    
    for (int i = 0; i < Mat->row; i++)
    {
        for(int j = 0; j < Mat->col; j++)
        {
            printf("%.2f\t", Mat->vals[i][j]);
        }
        printf("\n");
    }
    
    printf("\n\n");
}

void print_matrix_dim(Matrix *Mat)
{
    printf("Dim: %d x %d\n", Mat->row, Mat->col);
    printf("\n\n");
}

void free_matrix(Matrix *Mat)
{
    for(int i = 0; i < Mat->row; i++)
    {
        free(Mat->vals[i]);
    }
    
    free(Mat->vals);
}

Matrix copy_Mat_wCPU(Matrix *Mat_In)
{
    Matrix tempMat = create_matrix(Mat_In->row, Mat_In->col);
    
    for (int i = 0; i < Mat_In->row; i++)
    {
        for (int j = 0; j < Mat_In->col; j++)
        {
            tempMat.vals[i][j] = Mat_In->vals[i][j];
        }
    }
    
    return tempMat;
}

Matrix transpose_Mat_wCPU(Matrix *Mat_In)
{
    Matrix tempMat = create_matrix(Mat_In->col, Mat_In->row);
    
    for (int i = 0; i < Mat_In->row; i++)
    {
        for(int j = 0; j < Mat_In->col; j++)
        {
            tempMat.vals[j][i] = Mat_In->vals[i][j];
        }
    }
    
    return tempMat;
}

