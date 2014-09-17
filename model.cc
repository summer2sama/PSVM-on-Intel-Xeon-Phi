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

#include <cstring>
#include <iostream>
#include <vector>
#include <utility>
#include <string>
#include "omp.h"

#include "model.h"
#include "document.h"
#include "pd_ipm_parm.h"
#include "kernel.h"
#include "matrix.h"
#include "io.h"
#include "util.h"

namespace psvm {

Model::Model() {
	num_total_sv_ = 0;
}

Model::~Model() {
}

// Checks whether the alpha values (stores in double *alpha) are support
// vecotrs. And fills the support vector object (support_vector_) of
// class Model.
//
// If alpha[i] < epsilon_sv, then the i'th sample is not considered as a
// support vector. if (alpha[i] - C) is smaller than epsilon_sv, than
// alpha[i] is considered as a support vector and regulate as C. Any other
// alpha values between 0 and C will be considered as support vector.
//
// In the weighted case, in which positive and negative samples possess
// different weight, then C equals hyper_parm * weight.
void Model::CheckSupportVector(double *alpha,struct Document doc,int *Sample_label,
                               int *Sample_Feature_id, double *Sample_Feature_weight,double *Sample_two_norm_sq,
                               struct PrimalDualIPMParameter ipm_parameter) {

  int num_svs = 0;  // number of support vectors
  int num_bsv = 0;  // number of boundary support vectors
  int num_global_rows = doc.num_total_;  // number of samples
  // pos_sv[i] stores the index of the i'th support vector.
  int *pos_sv = new int[num_global_rows];

  // Get weighted c for positive and negative samples.
  double c_pos = ipm_parameter.hyper_parm * ipm_parameter.weight_positive;
  double c_neg = ipm_parameter.hyper_parm * ipm_parameter.weight_negative;
 
  // Check and regulate support vector values.
  for (int i = 0; i < num_global_rows; ++i) {
    if (alpha[i] <= ipm_parameter.epsilon_sv) {
      // If alpha[i] is smaller than epsilon_sv, then assign alpha[i] to be 0,
      // which means samples with small alpha values are considered as
      // non-support-vectors.
      alpha[i] = 0;
    } else {
      // If alpha[i] is near the weighted hyper parameter, than regulate
      // alpha[i] to be the weighted hyper parameter.
      pos_sv[num_svs++] = i;
      double c = (Sample_label[i] > 0) ? c_pos : c_neg;
      if ((c - alpha[i]) <= ipm_parameter.epsilon_sv) {
        alpha[i] = c;
        ++num_bsv;
      }
    }
  }
  
  // Store support vector information
  support_vector_.num_sv      = num_svs;
  support_vector_.num_bsv     = num_bsv;
  support_vector_.num_feature = doc.num_feature;
  support_vector_.sv_alpha = new double[num_svs];
  support_vector_.SV_label = new int[num_svs];
  support_vector_.SV_Feature_id = new int[num_svs * doc.num_feature];
  support_vector_.SV_Feature_weight = new double[num_svs * doc.num_feature];
  support_vector_.SV_two_norm_sq = new double[num_svs];
 
  for (int i = 0; i < num_svs; i++) {
    // sv_alpha stores the production of alpha[i] and label[i].
    // sv_alpha[i] = alpha[i] * label[i].
    if (Sample_label[pos_sv[i]] > 0) {
      support_vector_.sv_alpha[i] = alpha[pos_sv[i]];
    }
    if (Sample_label[pos_sv[i]] < 0) {
      support_vector_.sv_alpha[i] = -alpha[pos_sv[i]];
    }
  }

  if (doc.num_total_ != 0) {
    for (int i = 0; i < num_svs; i++) {
      support_vector_.SV_two_norm_sq[i] = Sample_two_norm_sq[pos_sv[i]];
      support_vector_.SV_label[i] = Sample_label[pos_sv[i]];
      for (int j = 0; j < doc.num_feature; j++){
      support_vector_.SV_Feature_id[i*doc.num_feature + j] = Sample_Feature_id[pos_sv[i]*doc.num_feature + j];
      support_vector_.SV_Feature_weight[i*doc.num_feature + j] = Sample_Feature_weight[pos_sv[i]*doc.num_feature + j];
      }
    }
  }

  num_total_sv_ = support_vector_.num_sv;
  if (ipm_parameter.verb >= 1) {
    cout << "Number of Support Vectors = " << num_total_sv_ << endl;
  }

  delete [] pos_sv;
}

// Saves model in the "str_directory/" directory.
// several files will be created.
//   model.header: stores general information, including:
//        kernel parameters.
//        the b value of the obtained SVM.
//        the number of support vectors.
//   model.sv: stores the support vectors
//            The format is as follows:
//        alpha_value [feature_id:feature_value, ...]
void Model::Save(const char* str_directory, const char* model_name) {
    cout << "========== Store Model ==========" << endl;

  // Save model header
  SaveHeader(str_directory, model_name);
  // Output support vector alpha and data
  SaveChunks(str_directory, model_name);
}

void Model::SaveHeader(const char* str_directory, const char* model_name) {
    char str_file_name[4096];
    // Create a file for storing model.header
    snprintf(str_file_name, sizeof(str_file_name),
             "%s/%s.header", str_directory, model_name);
    File* obuf = File::OpenOrDie(str_file_name, "w");

    // Output kernel parameters
    cout << "Storing " << model_name << ".header ... " << endl;
	obuf->WriteString(StringPrintf("%d", kernel_.kernel_type_));
    switch (kernel_.kernel_type_) {
      case LINEAR:
        obuf->WriteString(StringPrintf("\n"));
        break;
      case POLYNOMIAL:
        obuf->WriteString(StringPrintf(" %.8lf %.8lf %d\n",
                         kernel_.coef_lin_,
                         kernel_.coef_const_,
                         kernel_.poly_degree_));
        break;
      case GAUSSIAN:
        obuf->WriteString(StringPrintf(" %.8lf\n", kernel_.rbf_gamma_));
        break;
      case LAPLACIAN:
        obuf->WriteString(StringPrintf(" %.8lf\n", kernel_.rbf_gamma_));
        break;
      default:
        cerr << "Error: Unknown kernel_ function\n";
        exit(1);
    }
    // Output b value and number of support vectors
    obuf->WriteString(StringPrintf("%.8lf\n", support_vector_.b));
    obuf->WriteString(StringPrintf("%d\n", support_vector_.num_sv));
    obuf->WriteString(StringPrintf("%d\n", support_vector_.num_feature));
    CHECK(obuf->Flush());
    CHECK(obuf->Close());
    delete obuf;
}

void Model::SaveChunks(const char* str_directory, const char* model_name) {
  cout << "Storing " << model_name << ".sv" << endl;
  char str_file_name[4096];
  snprintf(str_file_name, sizeof(str_file_name),
           "%s/%s.sv", str_directory, model_name);
  File* obuf = File::OpenOrDie(str_file_name, "w");

  // Output the number of support vectors of this model.
  obuf->WriteLine(StringPrintf("%d", support_vector_.num_sv));
  for (int i = 0; i < support_vector_.num_sv; i++) {
    // first, alpha value with label
    obuf->WriteString(StringPrintf("%.8lf", support_vector_.sv_alpha[i]));
    // then, support vector sample
    for (size_t j = 0; j < support_vector_.num_feature; j++) {
      obuf->WriteString(
          StringPrintf(" %d:%.8lf",
                       support_vector_.SV_Feature_id[i*support_vector_.num_feature + j],
                       support_vector_.SV_Feature_weight[i*support_vector_.num_feature + j]));
    }
    obuf->WriteLine(string(""));
  }

  CHECK(obuf->Flush());
  CHECK(obuf->Close());
  delete obuf;
}

/*
// Loads model from the "str_directory/" directory.
// For processor # (here # can be 0, 1, 2, ..., etc), two files will be read
//   model.header: stores general information, including:
//        kernel parameters.
//        the b value of the obtained SVM.
//        the number of support vectors.
//   model.#: the support vectors corresponding to processor #.
//            The format is as follows:
//        alpha_value [feature_id:feature_value, ...]
void Model::Load(const char* str_directory, const char* model_name) {
  // Load model header file
  LoadHeader(str_directory, model_name);

  //if (mpi_->GetNumProcs() == num_chunks_) {
    // Read model chunks directly
    LoadChunks(str_directory, model_name);
  //} else {
  //  cerr << "The number of processes used to predict is different from the number of processes used to train" << endl;
  //  exit(1);
 // }
}

void Model::LoadHeader(const char*str_directory, const char* model_name) {
  char str_file_name[4096];
  // Load model header file
  snprintf(str_file_name, sizeof(str_file_name),
           "%s/%s.header", str_directory, model_name);
  string line;
  File *reader = File::OpenOrDie(str_file_name, "r");

  // Read kernel_ parameters
  KernelType kernel_type;
  int  poly_degree, int_kernel_type;
  double rbf_gamma;
  double coef_lin;
  double coef_const;

  reader->ReadLine(&line);
  const char *buf = line.c_str();
  SplitOneIntToken(&buf, " ", &int_kernel_type);
  kernel_type = static_cast<KernelType>(int_kernel_type);
  kernel_.kernel_type_ = kernel_type;
  switch (kernel_type) {
    case Kernel::LINEAR:
      break;
    case Kernel::POLYNOMIAL:
      // polynomial kernel
      SplitOneDoubleToken(&buf, " ", &coef_lin);
      SplitOneDoubleToken(&buf, " ", &coef_const);
      SplitOneIntToken(&buf, "\n", &poly_degree);

      kernel_.coef_lin_ = coef_lin;
      kernel_.coef_const_ = coef_const;
      kernel_.poly_degree_ = poly_degree;
      break;
    case Kernel::GAUSSIAN:
      // gaussian kernel
      SplitOneDoubleToken(&buf, "\n", &rbf_gamma);
      kernel_.rbf_gamma_ = rbf_gamma;
      break;
    case Kernel::LAPLACIAN:
      // laplacian kernel
      SplitOneDoubleToken(&buf, "\n", &rbf_gamma);
      kernel_.rbf_gamma_ = rbf_gamma;
      break;
    default:
      cerr << "Fatal Error: Unknown kernel_ function" << endl;
      exit(1);
  }
  // Read b
  reader->ReadLine(&line);
  buf = line.c_str();
  SplitOneDoubleToken(&buf, "\n", &(support_vector_.b));

  // Read total number of support vectors, and number of chunks
  reader->ReadLine(&line);
  buf = line.c_str();
  SplitOneIntToken(&buf, "\n", &num_total_sv_);
  SplitOneIntToken(&buf, "\n", &support_vector_.num_feature);
  //SplitOneIntToken(&buf, "\n", &num_chunks_);

  // Clean up
  reader->Close();
  delete reader;
}

void Model::LoadChunks(const char*str_directory, const char* model_name) {
  char str_file_name[4096];
  snprintf(str_file_name, sizeof(str_file_name),
           "%s/%s.sv", str_directory, model_name);
  const char* buf;

  // Check the total number of support vectors
  File *reader = File::OpenOrDie(str_file_name, "r");
  string line;
  reader->ReadLine(&line);
  buf = line.c_str();
  int num_global_sv = 0;
  //int num_local_sv = 0;
  SplitOneIntToken(&buf, "\n", &num_global_sv);
  support_vector_.num_sv = num_global_sv;
  //mpi_->Reduce(&num_local_sv, &num_global_sv, 1,
  //             MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  //if (mpi_->GetProcId() == 0) {
    if (num_global_sv != num_total_sv_) {
      cerr << model_name << ".sv do not compatible with "
                 << model_name << ".dat" << endl
                 << "Expected total #SV:" << num_total_sv_
                 << "Actual total #SV:" << num_global_sv;
      exit(1);
    }
  //}

  // Read model.sv into sv_data_test
  int num_actual_global_sv = 0;
  int i = 0;
  while (reader->ReadLine(&line)) {
    buf = line.c_str();
    double alpha;
    SplitOneDoubleToken(&buf, " ", &alpha);
    support_vector_.sv_alpha[i] = alpha;

    //support_vector_.sv_data_test.push_back(Sample());
    //Sample& sample = support_vector_.sv_data_test.back();

    sample.id = num_actual_global_sv;
    alpha > 0 ? sample.label = 1 : sample.label = -1;
    sample.two_norm_sq = 0;
    if (buf != NULL) {
      vector<pair<string, string> > kv_pairs;
      SplitStringIntoKeyValuePairs(string(buf), ":", " ", &kv_pairs);
      vector<pair<string, string> >::const_iterator pair_iter;
      for (pair_iter = kv_pairs.begin();
           pair_iter != kv_pairs.end();
           ++pair_iter) {
        Feature feature;
        feature.id = atoi(pair_iter->first.c_str());
        feature.weight = atof(pair_iter->second.c_str());
        sample.features.push_back(feature);
        sample.two_norm_sq += feature.weight * feature.weight;
      }
    }
    ++num_actual_global_sv;
  }

  // Check  of support vectors
  if (num_actual_global_sv != num_global_sv) {
    cerr << str_file_name << " is broken!"
               << "Expected #SV:" << num_global_sv
               << "\tActual #SV:" << num_actual_global_sv;
    exit(1);
  }

  // Print local support vector number
  cout << "#support_vector_:"
            << support_vector_.num_sv << endl;

  // Clean up
  CHECK(reader->Close());
  delete reader;
}
*/


// Compute b of the support vector machine.
// Theory:
//   f(x) = sum{alpha[i] * kernel(x, x_i)} + b
//   where x_i is the i'th support vector.
//   Therefore, b can be abtained by substituting x by a support vector.
//   In order to get a robust estimation, we estimate b a few times, and use
//   the average value as an optimal estimation.
//
// Implementation Details:
//   First, each machine provides a few support vectors, these support vectors
// are broadcasted to other machines. All the provided support vectors formed
// a "selected support vector dataset".
//   Second, each computer compute a local b
// value using its local support vectors (stored in local machine),
// that is: sum{alpha[k] * kernel(x,x_k)}
// where x_k is the k'th local support vector. Note that each support vector
// in the selected support vector dataset can obtain such a local b, thus a
// local b value array can be obtained.
//   Third, sum local b value arrays of different machines, a global b value
// array can be obtained, which equals: sum{alpha[i] * kernel(x, x_i)}
//   Finally, the #0 machine get the average b value, save it in
// support_vector_.b
void Model::ComputeB(PrimalDualIPMParameter& ipm_parameter) {
    cout << "========== Compute b ==========" << endl;

  // selected support_vector_ count in computing b.
  int num_selected_sv = std::min(num_total_sv_, 1000);
double *global_b = new double[num_selected_sv];
  for (int i = 0; i < num_selected_sv; i++) {
    // Get global b values
    double sum = 0;
    for (int k = 0; k < support_vector_.num_sv; k++) {
      sum += support_vector_.sv_alpha[k] *
          CalcKernel(kernel_,support_vector_.num_feature,support_vector_.SV_Feature_id,support_vector_.SV_Feature_weight, 
                     k,i,support_vector_.SV_two_norm_sq);
    }
    global_b[i] = support_vector_.SV_label[i] - sum;
  }

    double b = 0;
    //#pragma omp parallel for reduction(+:b)
    for (int i = 0; i < num_selected_sv; i++)
      b += global_b[i];
    b /= num_selected_sv;

    // Output b value.
    support_vector_.b = b;
    cout << "          b = "
              << support_vector_.b
              << endl;
  delete [] global_b;
}
}
