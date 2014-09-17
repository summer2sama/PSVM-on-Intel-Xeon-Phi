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

#ifndef MODEL_H__
#define MODEL_H__

#include <vector>
#include "kernel.h"

using namespace std;

namespace psvm {
struct Document;   
struct PrimalDualIPMParameter;  
struct SupportVector {
  int num_sv;           // number of support vectors
  int num_bsv;          // number of support vectors at boundary
  double b;             // b value of classification function in SVM model
  double *sv_alpha;        // the alpha values of the support vectors
  int num_feature;
  int *SV_label;
  int *SV_Feature_id;
  double *SV_Feature_weight;
  double *SV_two_norm_sq;
};

class Model {
 public:
  Model();
  virtual ~Model();

  // Uses alpha values to decide which samples are support vectors and stores their information.
  void CheckSupportVector(double *alpha,struct Document doc,int *Sample_lable,
                               int *Sample_Feature_id, double *Sample_Feature_weight,double *Sample_two_norm_sq,
                               struct PrimalDualIPMParameter ipm_parameter);

  struct SupportVector support_vector() { return support_vector_; }

  // Accessors to kernel_.
  struct Kernel kernel() { return kernel_; }
  void set_kernel( struct Kernel kernel) { kernel_ = kernel; }

  // Saves the model to the directory specified by str_directory.
  void Save(const char* str_directory, const char* model_name);
  void SaveHeader(const char* str_directory, const char* model_name);
  void SaveChunks(const char* str_directory, const char* model_name);

  //Loads the model from the directory specified by str_directory.
  //void Load(const char* str_directory, const char* model_name);
  //void LoadHeader(const char* str_directory, const char* model_name);
  //void LoadChunks(const char* str_directory, const char* model_name);

  // Computes the b value of the SVM's classification function.
  void ComputeB(PrimalDualIPMParameter& ipm_parameter);

 private:
  // kernel_ stores kernel type, kernel parameter information,
  // and calculates kernel function accordingly.
  struct Kernel kernel_;

  // The number of support vectors in all.
  int num_total_sv_;

  // support_vector_ stores support vector information.
  struct SupportVector support_vector_;
};
}

#endif
