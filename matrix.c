/****************************************************************************
 *                                                                          *
 * 	DeepC: Deep Learning/Machine Learning Inference Library written in C 	*
 * 																			*
 * 	Copyright (C) 2018 by Andriyanto Halim          						*
 *                                                                          *
 *  This program is free software: you can redistribute it and/or modify	*
 *  it under the terms of the GNU General Public License as published by	*
 *  the Free Software Foundation, either version 3 of the License, or		*
 *  (at your option) any later version.										*
 *                                                                          *
 *  This program is distributed in the hope that it will be useful,        	*
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of        	*
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          	*
 *  GNU Lesser General Public License for more details.                    	*
 *                                                                         	*
 *  You should have received a copy of the GNU Lesser General Public       	*
 *  License along with this program. If not, see							*
 *  <http://www.gnu.org/licenses/>. 										*
 * 																			*
 ****************************************************************************/
 
/**
 * @file matrix.c
 * @brief Source file on detailed implementation for matrix related functions
 *
 * Collection of functions for matrix operations
 * 
 * @author Andriyanto Halim
 * @date 16 May 2018
 * 
 * @todo 
 * 1. change matrix length from "int" to "unsigned int"
 * 
 * @bug No known bugs
 * 
 * @see https://en.wikipedia.org/wiki/Array_data_structure
 */
 
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

Matrix random_matrix_normalized(int Mat_Row, int Mat_Col)
{
    Matrix tempMat = create_matrix(Mat_Row, Mat_Col);
    
    for (int i = 0; i < Mat_Row; i++)
    {
        for (int j = 0; j < Mat_Col; j++)
        {
            *(tempMat.vals[i] + j) = (float)(rand())/RAND_MAX;
        }
    }
    
    return tempMat;
}

Matrix random_matrix_ranged(int Mat_Row, int Mat_Col, int Max_Val, int Min_Val)
{
    Matrix tempMat = create_matrix(Mat_Row, Mat_Col);
    
    for (int i = 0; i < Mat_Row; i++)
    {
        for (int j = 0; j < Mat_Col; j++)
        {
            *(tempMat.vals[i] + j) = (float)(((rand()) % (Max_Val - Min_Val + 1)) + Min_Val);
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

