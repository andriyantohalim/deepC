#include "pooling.h"

Matrix maxpooling_Mat_wCPU(Matrix *Mat_In, int filter_height, int filter_width, int stride)
{
    assert(filter_height == filter_width);
    assert(filter_height == stride || filter_width == stride);
   
    // Calculate the needed padding_2d
    int temprowpad = stride - (Mat_In->row%stride);
    int tempcolpad = stride - (Mat_In->col%stride);

	// Perform padding_2d on Input Matrix
    vpadding_2d_asymmetric_Mat_wCPU(Mat_In, temprowpad, tempcolpad);
    
    // Calculate MaxPooling Output Matrix dimension based on the padded Input Matrix
	int temprow = Mat_In->row/filter_height;
    int tempcol = Mat_In->col/filter_width;
            
    Matrix tempMat = create_matrix(temprow, tempcol);    
    
    // p and q are the row- and col- indices of the maxpooling output matrix
    int p = 0;
    
    // i and j are the row- and col- indices of the input matrix
    for (int i = 0; i < Mat_In->row; i+= stride)
    {        
        int q = 0;
        
        for (int j = 0; j < Mat_In->col; j+= stride)
        {
            // Initialize a temp variable to store max value at the beginning of maxpool stride
            float tempmax = 0.0f;
    
            // m and n are the row- and col- indices of inner loop within each maxpool stride to find the max value
            for(int m = 0; m < filter_height; m++)
            {
                for(int n = 0; n < filter_width; n++)
                {
                    if(Mat_In->vals[m+i][n+j] > tempmax)
                    {
                        tempmax = Mat_In->vals[m+i][n+j];
                    }                        
                }
            }
            
            // Assign tempmax value once stride is finished.
            tempMat.vals[p][q] = tempmax;    
            
            q++;
        }
        
        p++;
    }
    
    return tempMat;
}

Tensor maxpooling_Tsr_wCPU(Tensor *Tsr_In, int filter_height, int filter_width, int stride)
{
	assert(filter_height == filter_width);
    assert(filter_height == stride || filter_width == stride);
   
    // Calculate the needed padding_2d
    int temprowpad = stride - (Tsr_In->row%stride);
    int tempcolpad = stride - (Tsr_In->col%stride);

	// Perform padding_2d on each input matrix layer
    vpadding_2d_asymmetric_Tsr_wCPU(Tsr_In, temprowpad, tempcolpad);
    
    // Calculate MaxPooling Output Matrix dimension based on the padded Input Matrix
	int temprow = Tsr_In->row/filter_height;
    int tempcol = Tsr_In->col/filter_width;
            
    Tensor tempTsr = create_tensor(temprow, tempcol, Tsr_In->depth);    
    
    for(int k = 0; k < Tsr_In->depth; k++)
    {
		// p and q are the row- and col- indices of the maxpooling output matrix
		int p = 0;
		
		// i and j are the row- and col- indices of the input matrix
		for (int i = 0; i < Tsr_In->row; i+= stride)
		{        
			int q = 0;
			
			for (int j = 0; j < Tsr_In->col; j+= stride)
			{
				// Initialize a temp variable to store max value at the beginning of maxpool stride
				float tempmax = 0.0f;
		
				// m and n are the row- and col- indices of inner loop within each maxpool stride to find the max value
				for(int m = 0; m < filter_height; m++)
				{
					for(int n = 0; n < filter_width; n++)
					{
						if(Tsr_In->vals[k][m+i][n+j] > tempmax)
						{
							tempmax = Tsr_In->vals[k][m+i][n+j];
						}                        
					}
				}
				
				// Assign tempmax value once stride is finished.
				tempTsr.vals[k][p][q] = tempmax;    
				
				q++;
			}
			
			p++;
		}
	}
    
    return tempTsr;
}
