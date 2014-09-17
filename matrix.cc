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

#include <cmath>
#include <cstring>

#include "matrix.h"
#include "document.h"
#include "kernel.h"
#include "pd_ipm_parm.h"
#include "util.h"
#include "io.h"
using namespace std;

namespace psvm {
void ParallelMatrix_Destroy(double **element_, struct ParallelMatrix *parallelmatrix) {
  if (element_) {
    for (int i = 0; i < parallelmatrix->num_cols_; ++i) {
      delete[] element_[i];
    }
    delete[] element_;
    element_ = NULL;
    parallelmatrix->num_cols_ = 0;
    parallelmatrix->num_rows_ = 0;
  }
}

void ParallelMatrix_Init(double **element_, struct ParallelMatrix *parallelmatrix, int num_rows, int num_cols) {
  struct ParallelMatrix parallelmatrix_tmp;
  ParallelMatrix_Destroy(element_, &parallelmatrix_tmp);
  parallelmatrix->num_rows_ = num_rows;
  parallelmatrix->num_cols_ = num_cols;
  //ParallelInterface* pinterface = ParallelInterface::GetParallelInterface();
  //num_local_rows_ = pinterface->ComputeNumLocal(num_rows_);
  element_ = new double* [parallelmatrix->num_cols_];
  for (int i = 0; i < parallelmatrix->num_cols_; ++i) {
    element_[i] = new double[parallelmatrix->num_rows_];
    memset(element_[i], 0, sizeof(**element_) * parallelmatrix->num_rows_);
  }
}

void ParallelMatrix_Save(const char *path, const char* file_name, struct ParallelMatrix parallelmatrix, double ** element_) {
  char filename[4096];
  snprintf(filename, sizeof(filename), "%s/%s.icf", path, file_name);
  File* obuf = File::OpenOrDie(filename, "w");

  // Write header info
  cout << "Saving matrix header to " << filename << "... " << endl;
  //CHECK(Write(obuf, &num_procs, sizeof(num_procs)));
  CHECK(ParallelMatrix_Write(obuf, &parallelmatrix.num_rows_, sizeof(parallelmatrix.num_rows_)));
  CHECK(ParallelMatrix_Write(obuf, &parallelmatrix.num_cols_, sizeof(parallelmatrix.num_cols_)));

  cout << "Saving matrix content to " << filename <<"... " << endl;
  for (int i = 0; i < parallelmatrix.num_rows_; i++) {
    for (int j = 0; j < parallelmatrix.num_cols_; j++) {
      double h_ij = element_[j][i];
      CHECK(ParallelMatrix_Write(obuf, &h_ij, sizeof(h_ij)));
    }
  }
  cout << "done" << endl;

  CHECK(obuf->Flush());
  CHECK(obuf->Close());
  delete obuf;
}

bool ParallelMatrix_Load(const char *path, const char* file_name, struct ParallelMatrix parallelmatrix, double ** element_) {
  char filename[4096];
  cout << "Tring to load " << file_name << endl;
  snprintf(filename, sizeof(filename), "%s/%s.icf", path, file_name);
  File *file_0 = File::Open(filename, "rb");
  if (file_0 == NULL) {
      cout << "matrix chunck file\""
                << filename
                << "\" does not exist!"
                << endl;
    return false;
  }

  // Read header info
  int num_original_rows;
  int num_original_cols;
	  cout << "Reading header info ... " << endl;
  if (file_0->Size() < sizeof(num_original_rows) * 3) {
      cout << "matrix chunck file\""
                << filename
                << "\" is too small in file size, maybe have been damaged!"
                << endl;
    return false;
  } else {
    if (
        !ParallelMatrix_Read(file_0, &num_original_rows, sizeof(num_original_rows)) ||
        !ParallelMatrix_Read(file_0, &num_original_cols, sizeof(num_original_cols))) {
      CHECK(file_0->Close());
      delete file_0;
        cout << "Truncated matrix file '" << filename << "'. "
                  << "Failed to load" << endl;
      return false;
    }
  }
  CHECK(file_0->Close());
  delete file_0;
		cout << "Reading trunks ... " << endl;
    return ParallelMatrix_ReadChunks(path, file_name, parallelmatrix, element_);
}


// 1. Open chunk file
// 2. Check the file size
// 3. Initialize matrix
// 4. Read corresponding chunk
bool ParallelMatrix_ReadChunks(const char *path, const char* file_name, struct ParallelMatrix parallelmatrix, double ** element_) {
  // Open matrix chunks
  char filename[4096];
  snprintf(filename, sizeof(filename), "%s/%s.icf", path, file_name);
  File *file = File::Open(filename, "rb");
  if (file == NULL) {
      cout << "matrix chunck file \"" << filename
                << "\" does not exist" << endl;
    return false;
  }

  // Read chunk header.
  int num_rows;
  int num_cols;
  if (
      !ParallelMatrix_Read(file, &num_rows, sizeof(num_rows)) ||
      !ParallelMatrix_Read(file, &num_cols, sizeof(num_cols))) {
      cout << "matrix chunck file \"" << filename
                << "\" is too small in file size, maybe have been damaged!"
                << endl;
    return false;
  }
  long long num_file_size = sizeof(num_rows) * 3 +
      parallelmatrix.num_rows_ * parallelmatrix.num_cols_ * sizeof(element_[0][0]);
  if ((long long) file->Size() != num_file_size) {
    cout << "The size of matrix chunck file \""
              << filename
              << "\" is not correct!" << endl
              << " Expected Size: " << num_file_size
              << " Real Size:" << file->Size() << endl;
    return false;
  }


  // Initialize matrix
  ParallelMatrix_Init(element_, &parallelmatrix, num_rows, num_cols);

  // Read matrix chunk
  for (int i = 0; i < parallelmatrix.num_rows_; ++i) {
    for (int j = 0; j < parallelmatrix.num_cols_; ++j) {
      double h_ij;
      if (!ParallelMatrix_Read(file, &h_ij, sizeof(h_ij)))
        return false;
      element_[j][i] = h_ij;
    }
  }
  file->Close();
  delete[] file;

  return true;
}


bool ParallelMatrix_Read(File* inbuf, void* buf, size_t size) {
  return (inbuf->Read(buf, size) == size);
}

bool ParallelMatrix_Write(File* obuf, void* buf, size_t size) {
  return (obuf->Write(buf, size) == size);
}

void LLMatrix_Destroy(double ** element_,int dim) {
  if (element_ != NULL) {
    for (int i = 0; i < dim; ++i) {
      delete[] element_[i];
    }
    delete[] element_;
    element_ = NULL;
  }
}

void LLMatrix_Init(double ** element_,int dim) {
  LLMatrix_Destroy(element_,dim);
  element_ = new double*[dim];
  for (int i = 0; i < dim; ++i) {
    element_[i] = new double[dim-i];
    memset(element_[i], 0, sizeof(**element_) * (dim-i));
  }
}
}
