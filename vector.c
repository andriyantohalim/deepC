#include "vector.h"

Vector create_vector(int Vec_Len)
{
	Vector V;
	
	V.len = Vec_Len;
	
	V.vals = calloc(V.len, sizeof(float));
	
	return V;
}

Vector constant_vector(int Vec_Len, float val)
{
	Vector tempVec = create_vector(Vec_Len);
	
	for (int i = 0; i < Vec_Len; i++)
	{
		tempVec.vals[i] = val;
	}
	
	return tempVec;
}

Vector random_vector(int Vec_Len)
{
	Vector tempVec = create_vector(Vec_Len);
	
	for (int i = 0; i < Vec_Len; i++)
	{
		tempVec.vals[i] = 2*i+1.3;
	}
	
	return tempVec;
}

void print_vector(Vector *Vec)
{
	printf("Len: %d\n", Vec->len);
	
	for (int i = 0; i < Vec->len; i++)
	{
		printf("%.2f\t", Vec->vals[i]);
	}
	
	printf("\n\n");
}

void print_vector_dim(Vector *Vec)
{
	printf("Len: %d \n", Vec->len);
    printf("\n\n");
}

void free_vector(Vector *Vec)
{
	free(Vec->vals);
}

Vector copy_Vec_wCPU(Vector *Vec_In)
{
	Vector tempVec = create_vector(Vec_In->len);
	
	for (int i = 0; i < Vec_In->len; i++)
	{
		tempVec.vals[i] = Vec_In->vals[i];
	}
	
	return tempVec;
}






