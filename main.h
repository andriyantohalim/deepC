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
 
/*! \mainpage DeepC: Deep Learning Library written in C Language
 *
 * \section Introduction
 *
 * This is the introduction.
 *
 * \section Key Features
 * 
 * asdfasdfasdfasdfasdf
 *
 * 
 * \section Limitation
 * asdfasdfasdf
 * 
 * \subsection 1. asdfasdfaf
 * asdfasdasdf
 * 
 * \section Examples
 * asdfasdfasdf
 * 
 * \subsection 1. Vectors
 * 
 * asdfasdfasdasdf
 * @code
 * int a = 0;
 * @endcode
 * 
 * \subsection 2. Matrices
 * 
 * asdfasdfasdasdf
 * 
 * \subsection 3. Tensors
 * 
 * asdfasdfasdasdf
 * 
 * \subsection 4. Data Conversion
 * 
 * asdfasdfasdasdf
 * 
 * \subsection 5. LeNet example
 * 
 * asdfasdfasdasdf
 * 
 * \section Compilation
 * Run the following command:
 * @code
 * gcc *.c -std=c99 -o main -lm && ./main
 * @endcode
 */
 
/**
 * @file main.h
 * @brief Header file for main.c
 *
 * Here typically goes a more extensive explanation of what the header
 * defines. Doxygens tags are words preceeded by either a backslash @\
 * or by an at symbol @@.
 * 
 * @author Andriyanto Halim (andriyanto.halim@gmail.com)
 * @date 16 May 2018
 * 
 * @todo 
 * 1. Refer to each header files for todo list
 * 2. Create a mainpage for Doxygen documentation
 * 3. Update author's email on each page
 * 
 * @bug No known bugs
 */
 
#ifndef MAIN_H
#define MAIN_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vector.h"
#include "matrix.h"
#include "tensor.h"
#include "activation.h"
#include "blas.h"
#include "convolution.h"
#include "data_conversion.h"
#include "fully_connected.h"
#include "padding.h"
#include "pooling.h"
#include "statistics.h"

/**
 * @brief main() function declaration
 */
int main(void);

#endif /* MAIN_H */
