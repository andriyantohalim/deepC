#include "tensor.h"

Tensor create_tensor(int Tsr_Row, int Tsr_Col, int Tsr_Depth)
{
	Tensor T;
	
	T.row = Tsr_Row;
	T.col = Tsr_Col;
	T.depth = Tsr_Depth;
	
	T.vals = calloc(T.depth, sizeof(float **));
	
	for (int k = 0; k < T.depth; k++)
	{
		T.vals[k] = calloc(T.row, sizeof(float *));
		
		for (int i = 0; i < T.row; i++)
		{
			T.vals[k][i] = calloc(T.col, sizeof(float));
		}
	}
	
	return T;
}

Tensor constant_tensor(int Tsr_Row, int Tsr_Col, int Tsr_Depth, float val)
{
    Tensor tempTensor = create_tensor(Tsr_Row, Tsr_Col, Tsr_Depth);
    
    for (int k = 0; k < Tsr_Depth; k++)
    {
		for (int i = 0; i < Tsr_Row; i++)
		{
			for (int j = 0; j < Tsr_Col; j++)
			{
				tempTensor.vals[k][i][j] = val;
			}
		}
	}
    
    return tempTensor;
}

Tensor random_tensor(int Tsr_Row, int Tsr_Col, int Tsr_Depth)
{
    Tensor tempTensor = create_tensor(Tsr_Row, Tsr_Col, Tsr_Depth);
    
    for (int k = 0; k < Tsr_Depth; k++)
    {
		for (int i = 0; i < Tsr_Row; i++)
		{
			for (int j = 0; j < Tsr_Col; j++)
			{
				tempTensor.vals[k][i][j] = i + j + k + 2*i + 0.5*j + 0.3*k + 2.12f;
			}
		}
	}
    
    return tempTensor;
}

void print_tensor(Tensor *Tsr)
{
	printf("Matrix dim: %d x %d\n", Tsr->row, Tsr->col);
	printf("Layer size: %d\n", Tsr->depth);

	for(int k = 0; k < Tsr->depth; ++k)
	{
		printf("Matrix[%d]:\n", k);
		
		for (int i = 0; i < Tsr->row; ++i)
		{
			for(int j = 0; j < Tsr->col; ++j)
			{
				printf("%.2f\t", Tsr->vals[k][i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}
	
	printf("\n\n");
}

void print_tensor_dim(Tensor *Tsr)
{
	printf("Matrix dim: %d x %d\n", Tsr->row, Tsr->col);
	printf("Layer size: %d\n", Tsr->depth);
	printf("\n\n");
}

void free_tensor(Tensor *Tsr)
{
    for(int k = 0; k < Tsr->depth; k++)
    {
		for(int i = 0; i < Tsr->row; i++)
		{
			free(Tsr->vals[k][i]);
		}
		
		free(Tsr->vals[k]);
	}

    free(Tsr->vals);
}

Tensor copy_Tsr_wCPU(Tensor *Tsr_In)
{
    Tensor tempTsr = create_tensor(Tsr_In->row, Tsr_In->col, Tsr_In->depth);
    
    for(int k = 0; k < Tsr_In->depth; k++)
    {
		for (int i = 0; i < Tsr_In->row; i++)
		{
			for (int j = 0; j < Tsr_In->col; j++)
			{
				tempTsr.vals[k][i][j] = Tsr_In->vals[k][i][j];
			}
		}
	}
		
    return tempTsr;
}

