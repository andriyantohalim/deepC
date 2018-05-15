#include "main.h"
#include <time.h>

// Compilation and run command
// gcc *.c -std=c99 -o main -lm && ./main

int main(void)
{    
	clock_t start, end;
	
	double cpu_time;
	double fps_measured;
	
	// record currect clock as start point
	start = clock();
	
	Tensor AAA = random_tensor_normalized(448,448, 3);
	
	Tensor kkk0 = random_tensor_normalized(7,7, 3);
	Tensor BBB = convolution_2d_Tsr_wCPU(&AAA, &kkk0, 3, 5);
	Tensor BBBo = Leaky_ReLU_Tsr_wCPU(&BBB);
	
	Tensor kkk1 = random_tensor_normalized(5,5, 5);
	Tensor BBBn = normalization_Tsr_wCPU(&BBBo);
	Tensor CCC = convolution_2d_Tsr_wCPU(&BBBn, &kkk1, 2, 5);
	Tensor CCCo = Leaky_ReLU_Tsr_wCPU(&CCC);
	
	Tensor kkk2 = random_tensor_normalized(5,5, 5);
	Tensor CCCn = normalization_Tsr_wCPU(&CCCo);
	Tensor DDD = convolution_2d_Tsr_wCPU(&CCCn, &kkk2, 2, 4);
	Tensor DDDo = Leaky_ReLU_Tsr_wCPU(&DDD);
	
	Tensor kkk3 = random_tensor_normalized(3,3, 4);
	Tensor DDDn = normalization_Tsr_wCPU(&DDDo);
	Tensor EEE = convolution_2d_Tsr_wCPU(&DDDn, &kkk3, 1, 4);
	Tensor EEEo = Leaky_ReLU_Tsr_wCPU(&EEE);
	
	Tensor kkk4 = random_tensor_normalized(3,3, 4);
	Tensor EEEn = normalization_Tsr_wCPU(&EEEo);
	Tensor FFF = convolution_2d_Tsr_wCPU(&EEEn, &kkk4, 1, 5);
	Tensor FFFo = Leaky_ReLU_Tsr_wCPU(&FFF);
	
	Tensor kkk5 = random_tensor_normalized(3,3, 5);
	Tensor FFFn = normalization_Tsr_wCPU(&FFFo);
	Tensor GGG = convolution_2d_Tsr_wCPU(&FFFn, &kkk5, 2, 5);
	Tensor GGGo = Leaky_ReLU_Tsr_wCPU(&GGG);
	
	Tensor kkk6 = random_tensor_normalized(3,3, 5);
	Tensor GGGn = normalization_Tsr_wCPU(&GGGo);
	Tensor HHH = convolution_2d_Tsr_wCPU(&GGGn, &kkk6, 1, 10);
	Tensor HHHo = Leaky_ReLU_Tsr_wCPU(&HHH);
	
	Tensor kkk7 = random_tensor_normalized(3,3, 10);
	Tensor HHHn = normalization_Tsr_wCPU(&HHHo);
	Tensor III = convolution_2d_Tsr_wCPU(&HHHn, &kkk7, 1, 100);
	Tensor IIIo = Leaky_ReLU_Tsr_wCPU(&III);	
	
	Vector I = Tsr2Vec_wCPU(&IIIo);

	Matrix WW0 = random_matrix_normalized(600, I.len);
	Vector B0 = constant_vector(600, 0.000523f);
	 
	Vector J = Fully_Connected_wCPU(&I, &WW0, &B0);
	Vector Jo = sigmoid_Vec_wCPU(&J);
	
	Matrix WW1 = constant_matrix(30, 600, 0.000123f);
	Vector B1 = constant_vector(30, 0.006243f);
	
	Vector K = Fully_Connected_wCPU(&Jo, &WW1, &B1);
	Vector Ko = sigmoid_Vec_wCPU(&K);
	
	Matrix WW2 = constant_matrix(5, 30, 0.000415f);
	Vector B2 = random_vector_normalized(5);
	
	Vector L = Fully_Connected_wCPU(&Ko, &WW2, &B2);
	Vector Lo = softmax_Vec_wCPU(&L);
	
	print_vector(&Lo);
	
	// record currect clock as end point
	end = clock();
	
	// run time is end - start
	cpu_time = ((double)(end - start))/CLOCKS_PER_SEC;
	fps_measured = (double)(1/cpu_time);
	printf("CPU time used = %.6f sec\n", cpu_time);
	printf("Measured fps = %.2f fps\n", fps_measured);
	
	
	
	free_tensor(&AAA);
	
	free_tensor(&kkk0);
	free_tensor(&BBB);
	free_tensor(&BBBo);

	free_tensor(&kkk1);
	free_tensor(&BBBn);
	free_tensor(&CCC);
	free_tensor(&CCCo);
	
	free_tensor(&kkk2);
	free_tensor(&CCCn);
	free_tensor(&DDD);
	free_tensor(&DDDo);
	
	free_tensor(&kkk3);
	free_tensor(&DDDn);
	free_tensor(&EEE);
	free_tensor(&EEEo);	
	
	free_tensor(&kkk4);
	free_tensor(&EEEn);
	free_tensor(&FFF);
	free_tensor(&FFFo);
	
	free_tensor(&kkk5);
	free_tensor(&FFFn);
	free_tensor(&GGG);
	free_tensor(&GGGo);	
	
	free_tensor(&kkk6);
	free_tensor(&GGGn);
	free_tensor(&HHH);
	free_tensor(&HHHo);
	
	free_tensor(&kkk7);
	free_tensor(&HHHn);
	free_tensor(&III);
	free_tensor(&IIIo);
	
	free_vector(&I);
	
	free_matrix(&WW0);
	free_vector(&B0);
	
	free_vector(&J);
	free_vector(&Jo);
	
	free_matrix(&WW1);
	free_vector(&B1);
	
	free_vector(&K);
	free_vector(&Ko);
	
	free_matrix(&WW2);
	free_vector(&B2);	
	
	free_vector(&L);
	free_vector(&Lo);

	
	return 0;
}
