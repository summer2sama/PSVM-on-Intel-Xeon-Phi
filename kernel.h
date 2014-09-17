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

#ifndef KERNEL_H__
#define KERNEL_H__
namespace psvm {
enum KernelType { LINEAR = 0, POLYNOMIAL, GAUSSIAN, LAPLACIAN };
struct Kernel{
  // Keeps track of the kernel type of this kernel.
  KernelType kernel_type_;

  // The gamma parameter for the Gaussian and Laplacian kernel.
  // In Gaussian kernel
  //    k(a, b) = exp{-rbf_gamma_ * ||a - b||^2}
  // In Laplacian kernel
  //    k(a, b) = exp{-rbf_gamma_ * |a - b|}
  double rbf_gamma_;

  // The three parameters of the polynomial kernel, in which:
  //    k(a, b) = (coef_lin * a.b + coef_const_) ^ poly_degree_
  double coef_lin_;
  double coef_const_;
  int  poly_degree_;
};  

double CalcKernel(struct Kernel kernel,int num_feature, int *Sample_Feature_id, double *Sample_Feature_weight, int a_id, int b_id,double *Sample_two_norm_sq);

double CalcKernelWithLabel(struct Kernel kernel,int num_feature, int *Sample_Feature_id, double *Sample_Feature_weight, int a_id, int b_id,double *Sample_two_norm_sq, int *Sample_label);

double OneNormSub(int num_feature, int *Sample_Feature_id, double *Sample_Feature_weight, int a_id, int b_id);

double InnerProduct(int num_feature, int *Sample_Feature_id, double *Sample_Feature_weight, int a_id, int b_id);

}
#endif
