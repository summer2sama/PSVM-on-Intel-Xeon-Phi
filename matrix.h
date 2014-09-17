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

#ifndef MATRIX_H__
#define MATRIX_H__

#include "io.h"

namespace psvm {
struct PrimalDualIPMParameter;

// ICF Matrix H 
// ParallelMatrix: n*p parallel matrix. Data on computer is column based.

  // Destroys the matrix by freeing space of element_ if space has
  // been allocated.
  void ParallelMatrix_Destroy(double ** element_, struct ParallelMatrix *parallelmatrix);

  // Initilizes the matrix: original matrix is num_rows_*num_cols_
  void ParallelMatrix_Init(double ** element_, struct ParallelMatrix *parallelmatrix, int num_rows, int num_cols);


  void ParallelMatrix_Save(const char *path, const char* file_name, struct ParallelMatrix parallelmatrix, double ** element_);

  bool ParallelMatrix_Load(const char *path, const char* file_name, struct ParallelMatrix parallelmatrix, double ** element_);

  bool ParallelMatrix_ReadChunks(const char *path, const char* file_name, struct ParallelMatrix parallelmatrix, double ** element_);

  bool ParallelMatrix_Read(File* inbuf, void* buf, size_t size);

  bool ParallelMatrix_Write(File* obuf, void* buf, size_t size);

struct ParallelMatrix{
  int num_rows_;
  int num_cols_;
};

// LLMatrix: n*n lower triangular/symmetrical matrix.
// Storage: Because the matrix is either lower triangular or
//          symmentric, we only store lower half part of the matrix and the
//          storage is column based.
// Example: symmetric matrix (3*3) is
//            1 2 3
//            2 5 6
//            3 6 9
//            elements_: 1 2 3 / 5 6 / 9
//          lower triangular (3*3) is
//            1 0 0
//            2 5 0
//            3 6 9
//            elements_: 1 2 3 / 5 6 / 9

  // Destroy the matrix by freeing allocated space
  void LLMatrix_Destroy(double ** element_,int dim);

  void LLMatrix_Init(double ** element_,int dim);
//Used in ICF step
struct Pivot {
  double pivot_value;
  int pivot_index;
};
}

#endif
