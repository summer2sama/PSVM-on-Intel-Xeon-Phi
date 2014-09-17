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

#ifndef MATRIX_MANIPULATION_H__
#define MATRIX_MANIPULATION_H__

#include <string>

using namespace std;

namespace psvm {
struct PrimalDualIPMParameter;
struct Document;
struct Kernel;
struct ParallelMatrix;
struct Pivot;

// Provides static methods for manipulating matrix operations, including
// ICF (Incomplete Cholesky Factorization), CF (Cholesky Factorization),
// CholBackwardSub (solves A'x=b) and CholForwardSub (solves Ax=b).

  // Performs Cholesky factorization on 'symmetric_matrix'.
  // The result is a lower triangular matrix 'result', which satisfies
  //           symetric_matrix = result * result'
  //
 bool MatrixManipulation_CF(double * original,
                            double **low_triangular,int dim);

  // Performs incomplete Cholesky factorization incrementally on the kernel matrix.
  // ICF performs a incremental version of incomplete cholesky factorization.
  void MatrixManipulation_ICF(struct Document doc, struct Kernel kernel,int rows, int columns, double threshold,double **icf, 
                     struct ParallelMatrix *parallelmatrix,int *Sample_Feature_id, double *Sample_Feature_weight,
                     double *Sample_two_norm_sq, int *Sample_label);

  // Resolves a'x = b where a is lower triangular matrix, x and b are vectors.
  // It is easy to be solved by backward substitutions. 
  void MatrixManipulation_CholBackwardSub(double **a, double *b,
                                         double *x,int dim);

  // Resolves ax = b where a is lower triangular matrix, x and b are vectors.
  // It is easy to be solved by forward substitutions. 
  void MatrixManipulation_CholForwardSub(double **a, double *b,
                                        double *x,int dim);

}  // namespace psvm

#endif
