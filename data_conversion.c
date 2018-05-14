#include "data_conversion.h"

Vector Mat2Vec_wCPU(Matrix *Mat_In)
{
    int templen = Mat_In->row * Mat_In->col;  
    
    Vector tempVec = create_vector(templen);  
     
    for (int idx = 0; idx < tempVec.len; idx++)
    {  
        int idxrow = idx / Mat_In->col;
        int idxcol = idx % Mat_In->col;
        
        tempVec.vals[idx] = Mat_In->vals[idxrow][idxcol];
    }   
    
    return tempVec;
}

Vector Tsr2Vec_wCPU(Tensor *Tsr_In)
{
	int templen = Tsr_In->row * Tsr_In->col * Tsr_In->depth;
	
	Vector tempVec = create_vector(templen);
	
	for (int idx = 0; idx < tempVec.len; idx++)
	{
		int idxdepth = idx / (Tsr_In->row * Tsr_In->col);
		int idxrow = (idx / Tsr_In->col) % Tsr_In->row;
		int idxcol = idx % Tsr_In->col;
		
		tempVec.vals[idx] = Tsr_In->vals[idxdepth][idxrow][idxcol];
	}
	
	return tempVec;
}

Matrix Vec2Mat_wCPU(Vector *Vec_In, int Mat_Row, int Mat_Col)
{
	Matrix tempMat = create_matrix(Mat_Row, Mat_Col);

	int idx = 0;   
    
    for(int i = 0; i < tempMat.row; i++)
    {
        for(int j = 0; j < tempMat.col; j++)
        {          
			tempMat.vals[i][j] = Vec_In->vals[idx];
            
            idx++;
        }
    }
	   
    return tempMat;   
}

Tensor Vec2Tsr_wCPU(Vector *Vec_In, int Tsr_Row, int Tsr_Col, int Tsr_Depth)
{
	Tensor tempTsr = create_tensor(Tsr_Row, Tsr_Col, Tsr_Depth);
	
	int idx = 0;
	
	for(int k = 0; k < tempTsr.depth; k++)
	{
		for(int i = 0; i < tempTsr.row; i++)
		{
			for(int j = 0; j < tempTsr.col; j++)
			{
				tempTsr.vals[k][i][j] = Vec_In->vals[idx];
				
				idx++;
			}
		}
	}
	
	return tempTsr;
}

