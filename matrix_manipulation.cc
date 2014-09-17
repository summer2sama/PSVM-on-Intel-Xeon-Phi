/*
Copyright 2007 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
This file is changed by Yang Wu(summer2sama).
*/

#include <float.h>
#include <cmath>
#include <cstring>
#include "omp.h"

#include "matrix_manipulation.h"
#include "io.h"
#include "util.h"
#include "matrix.h"
#include "document.h"
#include "kernel.h"
#include "pd_ipm_parm.h"

namespace psvm {

// Cholesky factorization: factorize matrix A into LL^T.
// "original" represents A, low_triangular represents L.
bool MatrixManipulation_CF( double * original,
                            double **low_triangular,int dim) {
  for (int i = 0; i < dim; ++i) {
      for (int j = i; j < dim; ++j) {
          double sum = original[dim * i - i * (i + 1)/2 + j];
      //#pragma omp parallel for reduction(-:sum)
          for (int k = i-1; k >= 0; --k) {
              sum -= low_triangular[k][i - k] * low_triangular[k][j - k];
          }
          if (i == j) {
              if (sum <= 0) {  // sum should be larger than 0
              cerr << "Only symmetric positive definite matrix can perform"
                           " Cholesky factorization.";
              return false;
              }
          low_triangular[i][0] =  sqrt(sum);
          } else {
              low_triangular[i][j - i] =  sum/low_triangular[i][0];
          }
      }
  }
  return true;
}

//============================================= ICF =====================================================

void MatrixManipulation_ICF(struct Document doc, struct Kernel kernel,int rows, int columns, double threshold,double **icf, 
                     struct ParallelMatrix *parallelmatrix,int *Sample_Feature_id, double *Sample_Feature_weight,
                     double *Sample_two_norm_sq, int *Sample_label) {
  //CHECK(icf);
int* pivot = new int[columns];
bool* pivot_selected = new bool[rows];

  // diag1: the diagonal part of Q (the kernal matrix diagonal
  // diag2: the quadratic sum of a row of the ICF matrix
double* diag1 = new double[rows]; 
double* diag2 = new double[rows];
  for (int i = 0; i < rows; ++i) {
    diag1[i] = CalcKernelWithLabel(kernel,doc.num_feature,Sample_Feature_id,Sample_Feature_weight,i,i,Sample_two_norm_sq,Sample_label);
    diag2[i] = 0;
    pivot_selected[i] = 0;
  }
  double global_trace = 0;

  for (int column = 0; column < columns; column++) {
    for (int i = 0; i < rows; i++) {
      if (pivot_selected[i] == false) global_trace += diag1[i] - diag2[i];
	}

    //Find global pivot
    struct Pivot global_pivot;
    global_pivot.pivot_value = -DBL_MAX;
    global_pivot.pivot_index = -1;
double* tmp = new double[rows];
	for (int i = 0; i < rows; ++i) {
      tmp[i] = diag1[i] - diag2[i];
	}
	for (int i = 0; i < rows; ++i){
		if (pivot_selected[i] == false && tmp[i] > global_pivot.pivot_value) {
        global_pivot.pivot_index = i;
        global_pivot.pivot_value = tmp[i];
      }
	}
	delete []tmp;
  
    // Update pivot vector
    pivot[column] = global_pivot.pivot_index;

double* header_row = new double[column + 1];
    if (global_pivot.pivot_index != -1) {
      icf[column][global_pivot.pivot_index] = sqrt(global_pivot.pivot_value);// (6.15) H(i_k,k) = sqrt(v(i_k))
	  for (int j = 0; j <= column; ++j) {
        header_row[j] = icf[j][global_pivot.pivot_index];//header_row[j] = H(i_k,j)
      }

      pivot_selected[global_pivot.pivot_index] = true;
	} 

    // Calculate the column'th column
    // Note: 1. This order can improve numerical accuracy.
    //       2. Cache is used, will be faster too.
    for (int i = 0; i < rows; ++i) {
      if (pivot_selected[i] == false) {
        icf[column][i] = 0;
      }
	}
    for (int k = 0; k < column; ++k) {
      for (int i = 0; i < rows; ++i) {
        if (pivot_selected[i] == false) {
          icf[column][i] = icf[column][i] -
                          icf[k][i] * header_row[k];//H(j_k,k) = H(j_k,k) - H(j_k,k)*H(i_k,k)
        }
      }
    }
    for (int i = 0; i < rows; ++i) {
      if (pivot_selected[i] == false) {
        icf[column][i] =  icf[column][i] + 
CalcKernelWithLabel(kernel,doc.num_feature,Sample_Feature_id,Sample_Feature_weight,i,global_pivot.pivot_index,Sample_two_norm_sq,Sample_label);
          //H(j_k,k) = H(j_k,k) + Q(j_k,k)
      }                                                                       
    }
    for (int i = 0; i < rows; ++i) {
      if (pivot_selected[i] == false) {
        icf[column][i] = icf[column][i]/header_row[column];//H(j_k,k) = H(j_k,k) / H(i_k,k)
      }
    }

    for (int i = 0; i < rows; ++i) {
      diag2[i] += icf[column][i] * icf[column][i];
    }
    delete[] header_row;
  }
  delete[] pivot;
  delete[] pivot_selected;
  delete[] diag1;
  delete[] diag2;
}
//===================================== ICF End=========================================================

void MatrixManipulation_CholBackwardSub(double **a, double *b,
                                         double *x,int dim) {
  for (int k = dim - 1; k >= 0; --k) {
    double tmp = b[k];
    for (int i = k + 1; i < dim; ++i) {
      tmp -= x[i] * a[k][i - k];
    }
    x[k] = tmp / a[k][0];
  }
}

void MatrixManipulation_CholForwardSub(double **a, double *b,
                                        double *x,int dim) {
  for (int k = 0; k < dim; ++k) {
    double tmp = b[k];
    for (int i = 0; i < k; ++i) {
      tmp -= x[i] * a[i][k - i];
    }
    x[k] = tmp / a[k][0];
  }
}

}  // namespace psvm
