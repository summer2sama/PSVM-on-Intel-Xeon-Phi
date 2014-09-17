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
#include <string>
#include <cstdlib>
#include <cstdio>
#include "svm_train.h"
#include "timer.h"
#include "common.h"
#include "document.h"
#include "kernel.h"
#include "model.h"
#include "matrix.h"
#include "matrix_manipulation.h"
#include "pd_ipm.h"
#include "pd_ipm_parm.h"
#include "io.h"
#include "util.h"


namespace psvm {
void SvmTrainer::TrainModel(   struct Document doc,int *Sample_Feature_id, double *Sample_Feature_weight,
                            double *Sample_two_norm_sq, int *Sample_label,
                               struct Kernel kernel,
                               struct PrimalDualIPMParameter parameter,
                            Model* model, bool failsafe) {
  TrainingTimeProfile::icf.Start();
  int rank = static_cast<int>(doc.num_total_
                              * parameter.rank_ratio);

  double ** icf_result;
  struct ParallelMatrix parallelmatrix;
  
  parallelmatrix.num_rows_ = doc.num_total_;
  parallelmatrix.num_cols_ = rank;
  icf_result = new double* [parallelmatrix.num_cols_];
  for (int i = 0; i < parallelmatrix.num_cols_; ++i) {
    icf_result[i] = new double[parallelmatrix.num_rows_];
    memset(icf_result[i], 0, sizeof(**icf_result) * parallelmatrix.num_rows_);
  }
  
  if (failsafe) {
    int global_succeed = 0;
    if (ParallelMatrix_Load(parameter.model_path, "ICF",parallelmatrix,icf_result)) global_succeed = 1;
    if (global_succeed != 1) {
      // Performs Incomplete Cholesky Fatorization to
      // reduce the matrix dimension.
		  cout << "Performing ICF ... \t" << endl;
      Timer t1;
      t1.Start();
      MatrixManipulation_ICF(doc,kernel,doc.num_total_,rank,parameter.threshold,icf_result, 
                     &parallelmatrix,Sample_Feature_id,Sample_Feature_weight,Sample_two_norm_sq,Sample_label);
      t1.Stop();
        cout << "ICF completed in " << t1.total() << " seconds" << endl;
		cout << "Saving H ..." << endl;
      Timer t2;
      t2.Start();
      ParallelMatrix_Save(parameter.model_path, "ICF",parallelmatrix,icf_result);
      t2.Stop();
        cout << "H saved in " << t2.total() << " seconds" << endl;
	}
  } else {
	  cout << "Performing ICF ... " << endl;
    Timer t1;
    t1.Start();
    MatrixManipulation_ICF(doc,kernel,doc.num_total_,rank,parameter.threshold,icf_result, 
                     &parallelmatrix,Sample_Feature_id,Sample_Feature_weight,Sample_two_norm_sq,Sample_label);
    t1.Stop();
      cout << "ICF completed in " << t1.total() << " seconds" << endl;
  }
  TrainingTimeProfile::icf.Stop();

  // Performs the Primal-Dual Interior Point Method.
  TrainingTimeProfile::ipm.Start();
  PrimalDualIPM_Solve(parameter, icf_result, parallelmatrix, doc,Sample_label, Sample_Feature_id,
                   Sample_Feature_weight, Sample_two_norm_sq, model, failsafe);
  TrainingTimeProfile::ipm.Stop();
  ParallelMatrix_Destroy(icf_result,&parallelmatrix);
}

std::string SvmTrainer::PrintTimeInfo() {
  std::string str = "========== Training Time Statistics ==========\n";
  str += " Total                                    : "
         + TrainingTimeProfile::total.PrintInfo() + "\n";
  str += " 1. Read Document                         : "
         + TrainingTimeProfile::read_doc.PrintInfo() + "\n";
  str += " 2. Train Model                           : "
         + TrainingTimeProfile::train_model.PrintInfo() + "\n";
  str += "    2.1 ICF                               : "
         + TrainingTimeProfile::icf.PrintInfo() + "\n";
  str += "    2.2 IPM                               : "
         + TrainingTimeProfile::ipm.PrintInfo() + "\n";
  str += "        2.2.1 Compute Surrogate Gap       : "
         + TrainingTimeProfile::surrogate_gap.PrintInfo() + "\n";
  str += "        2.2.2 H * H^T * alpha             : "
         + TrainingTimeProfile::partial_z.PrintInfo() + "\n";
  str += "        2.2.3 E = I + H^T * D * H         : "
         + TrainingTimeProfile::production.PrintInfo() + "\n";
  str += "        2.2.4 CF                          : "
         + TrainingTimeProfile::cf.PrintInfo() + "\n";
  str += "        2.2.5 Compute and Update Variables: "
         + TrainingTimeProfile::update_variables.PrintInfo() + "\n";
  str += "        2.2.6 Check Stop Condition        : "
         + TrainingTimeProfile::check_stop.PrintInfo() + "\n";
  str += "        2.2.7 Check SV                    : "
         + TrainingTimeProfile::check_sv.PrintInfo() + "\n";
  str += "        2.2.8 Misc.                       : "
         + TrainingTimeProfile::ipm_misc.PrintInfo() + "\n";
  str += "    2.3 Estimate b                        : "
         + TrainingTimeProfile::compute_b.PrintInfo() + "\n";
  str += " 3. Store Model                           : "
         + TrainingTimeProfile::store_model.PrintInfo() + "\n";

  return str;
}

void SvmTrainer::SaveTimeInfo(const char *path, const char* file_name) {
  char filename[4096];
  snprintf(filename, sizeof(filename), "%s/%s.time", path, file_name);
  File* obuf = File::OpenOrDie(filename, "w");
  std::string str = PrintTimeInfo();
  CHECK(obuf->WriteString(str) == str.length());

  CHECK(obuf->Flush());
  CHECK(obuf->Close());
  delete obuf;
}
}

using namespace psvm;

//=============================================================================
// Parameter Definitions

// Incomplete Cholesky Fatorization related options
double FLAGS_fact_threshold = 1.0e-5;
double FLAGS_rank_ratio = 1.0;

// Kernel function related options
int FLAGS_kernel_type = 2;
double FLAGS_gamma = 1.0;
int FLAGS_poly_degree = 3;
double FLAGS_poly_coef = 1.0;
double FLAGS_poly_const = 1.0;

// Learning related options
double FLAGS_zero_threshold = 1.0e-9;
double FLAGS_sv_threshold = 1.0e-4;
double FLAGS_hyper_parm = 1.0;
double FLAGS_positive_weight = 1.0;
double FLAGS_negative_weight = 1.0;

// IPM related options
double FLAGS_feasible_threshold = 1.0e-3;
double FLAGS_surrogate_gap_threshold = 1.0e-3;
int FLAGS_max_iteration = 200;
double FLAGS_mu_factor = 10.0;

// Where to save the result
string FLAGS_model_path = ".";

// Miscs
bool FLAGS_verbose = false;
double FLAGS_save_interval = 600.0;
bool FLAGS_failsafe = false;
//=============================================================================

void Usage() {
  const char* msg =
      "svm_train: This program does the SVM trainings on the training samples and "
      "generages a SVM model for futures prediction use. Usage:\n"
      "  svm_train data_file\n"
      "\n"
      "  Flag descriptions:\n"
      "    -fact_threshold (When to stop ICF. Should be in (0, 1]) type: double\n"
      "      default: 1.0000000000000001e-05\n"
      "    -failsafe (Whether to enable failsafe feature.) type: bool default: false\n"
      "    -feasible_threshold (Necessary convergance conditions: primal residual <\n"
      "      feasible_threshold and dual residual < dual residual) type: double\n"
      "      default: 0.001\n"
      "    -gamma (Gamma value in Gaussian and Laplacian kernel) type: double\n"
      "      default: 1\n"
      "    -hyper_parm (Hyper-parameter C in SVM model) type: double default: 1\n"
      "    -kernel_type (Type of kernel function. Available types are: 0: Linear  1:\n"
      "      Polynomial  2: Gaussian  3: Laplacian) type: int32 default: 2\n"
      "    -max_iteration (Maximum iterations for the IPM method) type: int32\n"
      "      default: 200\n"
      "    -model_path (Where to save the resulting model) type: string default: .\n"
      "    -mu_factor (Increasing factor mu) type: double default: 10\n"
      "    -negative_weight (Set hyper-parameter for negative class to be\n"
      "      negative_weight * C) type: double default: 1\n"
      "    -poly_coef (Coefficient in Polynomial kernel) type: double default: 1\n"
      "    -poly_const (Constant in Polynomial kernel) type: double default: 1\n"
      "    -poly_degree (Degree in Polynomial kernel) type: int32 default: 3\n"
      "    -positive_weight (Set hyper-parameter for positive class to be\n"
      "      positive_weight * C) type: double default: 1\n"
      "    -rank_ratio (Ratio of rank to be used in ICF. Should be in (0, 1])\n"
      "      type: double default: 1\n"
      "    -save_interval (Number of seconds between two save operations) type: double\n"
      "      default: 600\n"
      "    -surrogate_gap_threshold (Necessary convergance condition: surrogate gap <\n"
      "      surrogate_gap_threshold) type: double default: 0.001\n"
      "    -sv_threshold (When to consider a variable as a support vector)\n"
      "      type: double default: 0.0001\n"
      "    -verbose (Whether to show additional information) type: bool default: false\n"
      "    -zero_threshold (When to consider a variable as zero) type: double\n"
      "      default: 1.0000000000000001e-09\n";
  cerr << msg;
}

void ParseCommandLine(int* argc, char*** argv) {
  int i;
  for (i = 1; i < *argc; ++i) {
    if ((*argv)[i][0] != '-') break;
    if (++i >= *argc) {
      Usage();
      exit(1);
    }

    char* param_name = &(*argv)[i-1][1];
    char* param_value = (*argv)[i];
    if (strcmp(param_name, "fact_threshold") == 0) {
      FLAGS_fact_threshold = atof(param_value);
    } else if (strcmp(param_name, "rank_ratio") == 0) {
      FLAGS_rank_ratio = atof(param_value);
    } else if (strcmp(param_name, "kernel_type") == 0) {
      FLAGS_kernel_type = atoi(param_value);
    } else if (strcmp(param_name, "gamma") == 0) {
      FLAGS_gamma = atof(param_value);
    } else if (strcmp(param_name, "poly_degree") == 0) {
      FLAGS_poly_degree = atoi(param_value);
    } else if (strcmp(param_name, "poly_coef") == 0) {
      FLAGS_poly_coef = atof(param_value);
    } else if (strcmp(param_name, "poly_const") == 0) {
      FLAGS_poly_const = atof(param_value);
    } else if (strcmp(param_name, "zero_threshold") == 0) {
      FLAGS_zero_threshold = atof(param_value);
    } else if (strcmp(param_name, "sv_threshold") == 0) {
      FLAGS_sv_threshold = atof(param_value);
    } else if (strcmp(param_name, "hyper_parm") == 0) {
      FLAGS_hyper_parm = atof(param_value);
    } else if (strcmp(param_name, "positive_weight") == 0) {
      FLAGS_positive_weight = atof(param_value);
    } else if (strcmp(param_name, "negative_weight") == 0) {
      FLAGS_negative_weight = atof(param_value);
    } else if (strcmp(param_name, "feasible_threshold") == 0) {
      FLAGS_feasible_threshold = atof(param_value);
    } else if (strcmp(param_name, "surrogate_gap_threshold") == 0) {
      FLAGS_surrogate_gap_threshold = atof(param_value);
    } else if (strcmp(param_name, "max_iteration") == 0) {
      FLAGS_max_iteration = atoi(param_value);
    } else if (strcmp(param_name, "mu_factor") == 0) {
      FLAGS_mu_factor = atof(param_value);
    } else if (strcmp(param_name, "model_path") == 0) {
      FLAGS_model_path = string(param_value);
    } else if (strcmp(param_name, "verbose") == 0) {
      if (strcmp(param_value, "false") != 0 && strcmp(param_value, "0") != 0) {
        FLAGS_verbose = true;
      }
    } else if (strcmp(param_name, "save_interval") == 0) {
      FLAGS_save_interval = atof(param_value);
    } else if (strcmp(param_name, "failsafe") == 0) {
      if (strcmp(param_value, "false") != 0 && strcmp(param_value, "0") != 0) {
        FLAGS_failsafe = true;
      }
    } else {
      cerr << "Unknown parameter " << param_name << endl;
      Usage();
      exit(2);
    }
  }

  for (int j = 1; i + j - 1 < *argc; ++j) {
    (*argv)[j] = (*argv)[i + j - 1];
  }
  *argc -= i - 1;
}

int main(int argc, char** argv) {
  ParseCommandLine(&argc, &argv);
struct Kernel kernel;
SvmTrainer trainer;
Model model;
struct Document doc;
PrimalDualIPMParameter ipm_parameter;

  // Verifies the options
  if ((FLAGS_fact_threshold <= 0 || FLAGS_fact_threshold > 1) ||
      (FLAGS_rank_ratio <=0 || FLAGS_rank_ratio > 1) ||
      (FLAGS_kernel_type < 0 || FLAGS_kernel_type > 3) ||
      (FLAGS_zero_threshold <= 0) ||
      (FLAGS_sv_threshold <= 0) ||
      (FLAGS_hyper_parm <= 0) ||
      (FLAGS_positive_weight <= 0 || FLAGS_negative_weight <= 0) ||
      (FLAGS_feasible_threshold < 0 || FLAGS_surrogate_gap_threshold < 0) ||
      (FLAGS_max_iteration < 1) ||
      (FLAGS_save_interval <= 0.0) ) {
    Usage();
    return 3;
  }

  // If no data file is specified
  if (argc != 2) {
    Usage();
    return 4;
  }

  // Sets the commandline options
  ipm_parameter.threshold = FLAGS_fact_threshold;
  ipm_parameter.rank_ratio = FLAGS_rank_ratio;
  kernel.kernel_type_ = static_cast<KernelType>(FLAGS_kernel_type);
  kernel.rbf_gamma_ = FLAGS_gamma;
  kernel.poly_degree_ = FLAGS_poly_degree;
  kernel.coef_lin_ = FLAGS_poly_coef;
  kernel.coef_const_ = FLAGS_poly_const;
  ipm_parameter.epsilon_x = FLAGS_zero_threshold;
  ipm_parameter.epsilon_sv = FLAGS_sv_threshold;
  ipm_parameter.hyper_parm = FLAGS_hyper_parm;
  strncpy(ipm_parameter.model_path, FLAGS_model_path.c_str(),
          sizeof(ipm_parameter.model_path));
  ipm_parameter.weight_positive = FLAGS_positive_weight;
  ipm_parameter.weight_negative = FLAGS_negative_weight;
  ipm_parameter.feas_thresh = FLAGS_feasible_threshold;
  ipm_parameter.sgap = FLAGS_surrogate_gap_threshold;
  ipm_parameter.max_iter = FLAGS_max_iteration;
  ipm_parameter.mu_factor = FLAGS_mu_factor;
  ipm_parameter.verb = FLAGS_verbose;
  ipm_parameter.save_interval = FLAGS_save_interval;
  ipm_parameter.tradeoff = 0;

  // Begin timing
  TrainingTimeProfile::total.Start();

  // Reads the training samples
  TrainingTimeProfile::read_doc.Start();
    cout << "Reading data ... " << endl;
  string data_file = argv[1];
    cout << "data file:" << data_file << endl;
  if (!Document_Count(data_file.c_str(),&doc)) {
    cerr << "Error reading file: " << data_file << endl;
    return 3;
  }
    cout << "Total: " << doc.num_total_
              << "  Positive: " << doc.num_pos_
              << "  Negative: " << doc.num_neg_ << endl;
  
int *Sample_label = new int[doc.num_total_];
double *Sample_two_norm_sq = new double[doc.num_total_];
int *Sample_Feature_id = new int[doc.num_total_ * doc.num_feature];
double *Sample_Feature_weight = new double[doc.num_total_ * doc.num_feature];

memset(Sample_two_norm_sq, 0, sizeof(*Sample_two_norm_sq) * (doc.num_total_));
memset(Sample_Feature_id, -1, sizeof(*Sample_Feature_id) * (doc.num_total_ * doc.num_feature));
memset(Sample_Feature_weight, 0, sizeof(*Sample_Feature_weight) * (doc.num_total_ * doc.num_feature));

  if (!Document_Read(data_file.c_str(),doc,Sample_label,Sample_two_norm_sq,Sample_Feature_id,Sample_Feature_weight)) {
    cerr << "Error reading file: " << data_file << endl;
    return 3;
  }
  TrainingTimeProfile::read_doc.Stop();

  // Performs PD-IPM SVM training based on ICF
  TrainingTimeProfile::train_model.Start();
  trainer.TrainModel(doc,Sample_Feature_id,Sample_Feature_weight,Sample_two_norm_sq,Sample_label,
                    kernel,ipm_parameter, &model, FLAGS_failsafe);

  // Based on the training result, computes the b value of the classification
  // function to complete the resulting model
  TrainingTimeProfile::compute_b.Start();
  model.set_kernel(kernel);
  model.ComputeB(ipm_parameter);
  TrainingTimeProfile::compute_b.Stop();
  TrainingTimeProfile::train_model.Stop();

  // Save the model.
  TrainingTimeProfile::store_model.Start();
  model.Save(ipm_parameter.model_path, "model");
  TrainingTimeProfile::store_model.Stop();

  // Stop timing
  TrainingTimeProfile::total.Stop();

  // Save time info
    cout << "Saving training time statistic info ... " << endl;
   trainer.SaveTimeInfo(ipm_parameter.model_path, "TrainingTimeInfo");
    cout << endl << trainer.PrintTimeInfo() << endl;
  return 0;
}
