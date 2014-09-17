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

#ifndef PD_IPM_H__
#define PD_IPM_H__

namespace psvm {
struct PrimalDualIPMParameter;
class ParallelMatrix;
class Document;
class Model;
class LLMatrix;

  // Using Newton method to solve the optimization problem
  int PrimalDualIPM_Solve (struct PrimalDualIPMParameter parameter,
                          double ** rbicf,struct ParallelMatrix parallelmatrix,
                          struct Document doc,int *Sample_lable,
                         int *Sample_Feature_id, double *Sample_Feature_weight,double *Sample_two_norm_sq,
                         Model* model,
                         bool failsafe);

  // Compute $HH^T\alpha$, which is part of $z$, $\alpha$ is primal variable
  int PrimalDualIPM_ComputePartialZ( double ** icf,struct ParallelMatrix parallelmatrix,
                                    double *x,  double to,
                                    int num_doc_rows,
                                   double *z);

  // Compute surrogate gap
  double PrimalDualIPM_ComputeSurrogateGap(double c_pos,
                                        double c_neg,
                                        double *label,
                                        int num_doc_rows,
                                        double *x,
                                        double *la,
                                        double *xi);

  // Compute direction of primal vairalbe $x$
  int PrimalDualIPM_ComputeDeltaX( double ** icf,struct ParallelMatrix parallelmatrix,
                                  double *d,  double *label,
                                  double dnu,  double ** lra, int lra_dim,
                                  double *z, int num_doc_rows,
                                 double *dx);

  // Compute direction of primal varialbe $\nu$
  int PrimalDualIPM_ComputeDeltaNu( double ** icf,struct ParallelMatrix parallelmatrix,
                                   double *d,  double *label,
                                   double *z,  double *x,
                                   double ** lra, int lra_dim, int num_doc_rows,
                                  double *dnu);

  // Solve a special form of linear equation using
  // Sherman-Morrison-Woodbury formula
  int PrimalDualIPM_LinearSolveViaICFCol( double **  icf,struct ParallelMatrix parallelmatrix,
                                         double *d,
                                        double *b,
                                         double ** lra,int lra_dim,
                                        int num_doc_rows,
                                        double *x);

  // Loads the values of alpha, xi, lambda and nu to resume from an interrupted
  // solving process.
  void PrimalDualIPM_LoadVariables(
     struct PrimalDualIPMParameter parameter,
    int num_total_doc, int *step,
    double *nu, double *x, double *la, double *xi);

  // Saves the values of alpha, xi, lambda and nu.
  void PrimalDualIPM_SaveVariables(
     struct PrimalDualIPMParameter parameter,
    int num_total_doc, int step,
    double nu, double *x, double *la, double *xi);
}  // namespace psvm

#endif
