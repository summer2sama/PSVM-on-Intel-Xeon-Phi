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

#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include "kernel.h"
#include "document.h"

namespace psvm {

// Computes the kernel function value for samples a and b according
// to kernel_type_. 
double CalcKernel(struct Kernel kernel,int num_feature, int *Sample_Feature_id, double *Sample_Feature_weight, int a_id, int b_id,double *Sample_two_norm_sq) {
  switch (kernel.kernel_type_) {
    case LINEAR:
      return InnerProduct(num_feature, Sample_Feature_id, Sample_Feature_weight, a_id, b_id) / sqrt(Sample_two_norm_sq[a_id] * Sample_two_norm_sq[b_id]);

    case POLYNOMIAL:
      double val, a_normalizer, b_normalizer;
      val = pow(kernel.coef_lin_ * InnerProduct(num_feature, Sample_Feature_id, Sample_Feature_weight, a_id, b_id) + kernel.coef_const_,
                static_cast<double>(kernel.poly_degree_));
      a_normalizer = pow(kernel.coef_lin_ * Sample_two_norm_sq[a_id] + kernel.coef_const_,
                         static_cast<double>(kernel.poly_degree_));
      b_normalizer = pow(kernel.coef_lin_ * Sample_two_norm_sq[b_id] + kernel.coef_const_,
                         static_cast<double>(kernel.poly_degree_));
      return val / sqrt(a_normalizer * b_normalizer);

    case GAUSSIAN:
      // Note: ||a - b||^2 = (||a||^2 - 2 * a.b + ||b||^2)
      return exp(- kernel.rbf_gamma_ *
                 (Sample_two_norm_sq[a_id] - 2 * InnerProduct(num_feature, Sample_Feature_id, Sample_Feature_weight, a_id, b_id) + Sample_two_norm_sq[b_id]));

    case LAPLACIAN:
      return exp(- kernel.rbf_gamma_ * OneNormSub(num_feature, Sample_Feature_id, Sample_Feature_weight, a_id, b_id));

    default:
     cerr <<"Unknown kernel type"<<endl;
      exit(1);
      return 0.0;  // This should not be executed because of LOG(FATAL)
  }
}

double CalcKernelWithLabel(struct Kernel kernel,int num_feature, int *Sample_Feature_id, double *Sample_Feature_weight, int a_id, int b_id,double *Sample_two_norm_sq, int *Sample_label) {
  return CalcKernel(kernel, num_feature, Sample_Feature_id, Sample_Feature_weight, a_id, b_id, Sample_two_norm_sq) * (Sample_label[a_id] == Sample_label[b_id] ? 1.0 : -1.0);
}


double OneNormSub(int num_feature, int *Sample_Feature_id, double *Sample_Feature_weight, int a_id, int b_id) {
  double norm = 0.0;
  int it1 = num_feature * a_id;
  int it2 = num_feature * b_id;
  
  // a.features and b.features must bemsorted according to the feature-id in the same order. 
  //We relies on this property to speed up the computation.
  while  ((it1 != num_feature * (a_id + 1)) && (it2 != num_feature * (b_id + 1))) {
    if (Sample_Feature_id[it1] == Sample_Feature_id[it2]) {
      norm += fabs(Sample_Feature_weight[it1] - Sample_Feature_weight[it2]);
      ++it1;
      ++it2;
    } else if (Sample_Feature_id[it1] < Sample_Feature_id[it2]) {
      norm += fabs(Sample_Feature_weight[it1]);
      ++it1;
    } else {
      norm += fabs(Sample_Feature_weight[it2]);
      ++it2;
    }
  }
  while (it1 != num_feature * (a_id + 1)) {
      norm += fabs(Sample_Feature_weight[it1]);
      ++it1;
  }
  while (it2 != num_feature * (b_id + 1)) {
      norm += fabs(Sample_Feature_weight[it2]);
      ++it2;
  }
  return norm;
}

double InnerProduct(int num_feature, int *Sample_Feature_id, double *Sample_Feature_weight, int a_id, int b_id) {
  double norm = 0.0;
  int it1 = num_feature * a_id;
  int it2 = num_feature * b_id;
  while ((it1 != num_feature * (a_id + 1)) && (it2 != num_feature * (b_id + 1))) {
    if (Sample_Feature_id[it1] == Sample_Feature_id[it2]) {
      norm += Sample_Feature_weight[it1] * Sample_Feature_weight[it2];
      ++it1;
      ++it2;
    } else if (Sample_Feature_id[it1] < Sample_Feature_id[it2]) {
      ++it1;
    } else {
      ++it2;
    }
  }
  return norm;
}
}
