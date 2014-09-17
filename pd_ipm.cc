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


#include <cfloat>
#include <cstring>
#include <vector>
#include <cmath>
#include <climits>
#include "omp.h"
#include "pd_ipm.h"
#include "timer.h"
#include "common.h"
#include "document.h"
#include "kernel.h"
#include "pd_ipm_parm.h"
#include "model.h"
#include "matrix.h"
#include "matrix_manipulation.h"
#include "util.h"
#include "io.h"
namespace psvm {


// The primal dual interior point method is based on the book below,
// Convex Optimization, Stephen Boyd and Lieven Vandenberghe,
// Cambridge University Press.
int PrimalDualIPM_Solve(   struct PrimalDualIPMParameter parameter,
                            double ** rbicf,struct ParallelMatrix parallelmatrix,
                            struct Document doc,int *Sample_label,
                         int *Sample_Feature_id, double *Sample_Feature_weight,double *Sample_two_norm_sq,
                         Model* model,
                         bool failsafe) {
  TrainingTimeProfile::ipm_misc.Start();
  register int i, step;
  int num_doc_rows = doc.num_total_;
  double c_pos = parameter.weight_positive * parameter.hyper_parm;
  double c_neg = parameter.weight_negative * parameter.hyper_parm;
  // Calculate total constraint
  //
  // Note  0 <= \alpha <= C, transform this
  // to 2 vector inequations, -\alpha <= 0 (code formulae 1) and
  // \alpha - C <= 0 (code formulae 2), this is standard
  // constraint form, refer to Convex
  // Optimiztion. so total inequality constraints is 2n.
  // note, n is length of vector \alpha.
  int num_constraints = num_doc_rows + num_doc_rows;

  // Allocating memory for variables
  //
  // Note x here means \alpha in paper. la here is
  // Langrange multiplier of (code formulae 2), \lambda in WG's paper.
  // xi here is Langrange multiplier of (code formulae 1), \xi in WG's paper.
  // nu here is Langrange multiplier of equality constraints, \nu in WG's
  // paper, here comes a little bit of explanation why \nu is a scalar instead
  // of a vector. Note, the equality constraint
  // coeffient matrix but 1-dim y^T,
  // substitute it to A in Convex Optimiztion (11.54), we know that \nu is a
  // scalar.
double *x = new double[num_doc_rows];
double *la = new double[num_doc_rows];
double *xi = new double[num_doc_rows];
double *label = new double[num_doc_rows];
  double label_temp;
  for (i = 0; i < num_doc_rows; ++i) {
    label_temp = static_cast<double>(Sample_label[i]);
    label[i] = label_temp;
  }
  double nu = 0.0;

  // tlx, tux here are temporary vectors, used to store intermediate result.
  // Actually, tlx stores \frac{1}{t \alpha},
  // tux stores \frac{1}{t(C - \alpha)}
  // refer to WG's paper formula (16) and (17).
  //
  // xilx, laux here are also temporary vectors.
  // xilx stores \frac{\xi}{\alpha}, laux stores \frac{\lambda}{C-\alpha}.
  //
  // Note all the division of vectors above is elements-wise division.
double *tlx = new double[num_doc_rows];
double *tux = new double[num_doc_rows];
double *xilx = new double[num_doc_rows];
double *laux = new double[num_doc_rows];

  // dla, dxi, dx, dnu are \lamba, \xi, \alpha, \nu in the Newton Step,
  // Note dnu is a scalar, all the other are vectors.
double *dla = new double[num_doc_rows];
double *dxi = new double[num_doc_rows];
double dnu = 0.0;
double *dx = new double[num_doc_rows];

  // d is the diagonal matrix,
  //   \diag(\frac{\xi_i}{\alpha_i} + \frac{\lambda_i}{C - \alpha_i}).
  //
  // Note in the code, z has two
  // phase of intue, the first result
  // is Q\alpha + 1_n + \nu y, part of formulae
  // (8) and (17), the last phase is to complete formulae (17)
double *d = new double[num_doc_rows];
double *z = new double[num_doc_rows];

  double t;     // step
  double eta;   // surrogate gap
  double resp;  // primal residual
  double resd;  // dual residual

  // initializes the primal-dual variables
  // last \lambda, \xi to accelerate Newton method.

  // initializes \lambda, \xi and \nu
  //   \lambda = \frac{C}{10}
  //   \xi = \frac{C}{10}
  //   \nu = 0

  nu = 0.0;
  memset(x, 0, sizeof(x[0]) * num_doc_rows);
double *c = new double[num_doc_rows]; 
  for (i = 0; i < num_doc_rows; ++i) {
    c[i] = (label[i] > 0) ? c_pos : c_neg;
    la[i] = c[i] / 10.0;
    xi[i] = c[i] / 10.0;
  }
  int rank = parallelmatrix.num_cols_;
  cout << StringPrintf("Training SVM ... (H = %d x %d)\n",
              num_doc_rows, rank);

  // Note icfA is p \times p Symetric Matrix, actually is I + H^T D H, refer
  // to WG's paper 4.3.2. We should compute (I + H^T D H)^{-1}, using linear
  // equations trick to get it later.

  double *icf_1 = new double[parallelmatrix.num_rows_ * parallelmatrix.num_cols_];
  for(int c1 = 0; c1 < parallelmatrix.num_cols_; c1 ++){
     for(int c2 = 0; c2 < parallelmatrix.num_rows_; c2 ++) {
       icf_1[c1 * parallelmatrix.num_rows_ + c2] = rbicf[c1][c2];
     }
  }

   int icfa_dim = parallelmatrix.num_cols_;
double *icfa_1 = new double [icfa_dim * (icfa_dim + 1) / 2];
memset(icfa_1, 0, sizeof(*icfa_1) * (icfa_dim * (icfa_dim + 1) / 2));

  double ** lra = new double*[icfa_dim];
  for (int i = 0; i < icfa_dim; ++i) {
    lra[i] = new double[icfa_dim-i];
    memset(lra[i], 0, sizeof(**lra) * (icfa_dim-i));
  }

  // iterating IPM algorithm based on ICF
  TrainingTimeProfile::ipm_misc.Stop();

  TrainingTimeProfile::ipm_misc.Start();
  step = 0;
  if (failsafe) {
    PrimalDualIPM_LoadVariables(parameter, num_doc_rows,
                  &step, &nu, x, la, xi);
  }
  double time_last_save = Timer::GetCurrentTime();
  TrainingTimeProfile::ipm_misc.Stop();
  int row = parallelmatrix.num_rows_;
  int column = parallelmatrix.num_cols_;
double *buff = new double[row];
  #pragma offload target(mic) in(icf_1:length(row * column) alloc_if(1) free_if(0)) nocopy(buff:length(row) alloc_if(1) free_if(0)) nocopy(icfa_1:length(icfa_dim * (icfa_dim + 1) / 2) alloc_if(1) free_if(0)) nocopy(d:length(row) alloc_if(1) free_if(0)) 
{}

//=====================================for step ==================================
  for (; step < parameter.max_iter; ++step) {
    TrainingTimeProfile::ipm_misc.Start();
    double time_current = Timer::GetCurrentTime();
    if (failsafe && time_current - time_last_save > parameter.save_interval) {
      PrimalDualIPM_SaveVariables(parameter, num_doc_rows,
                    step, nu, x, la, xi);
      time_last_save = time_current;
    }

      cout << StringPrintf("========== Iteration %d ==========\n", step);
    TrainingTimeProfile::ipm_misc.Stop();
    // Computing surrogate Gap
    // compute surrogate gap, for definition detail, refer to formulae (11.59)
    // in Convex Optimization. Note t and eta
    // have a relation, for more details,
    // refer to Algorithm 11.2 step 1. in Convext Optimization.
    TrainingTimeProfile::surrogate_gap.Start();
    eta = PrimalDualIPM_ComputeSurrogateGap(c_pos, c_neg, label, num_doc_rows, x, la, xi);
    // Note m is number of total constraints
    double t_temp = static_cast<double>(num_constraints);
    t = (parameter.mu_factor) * t_temp/ eta;

    if (parameter.verb >= 1) {
      cout << StringPrintf("sgap: %-.10le t: %-.10le\n", eta, t);
    }
    TrainingTimeProfile::surrogate_gap.Stop();


    // Check convergence
    // computes z = H H^T \alpha - tradeoff \alpha
    TrainingTimeProfile::partial_z.Start();
    PrimalDualIPM_ComputePartialZ(rbicf, parallelmatrix, x, parameter.tradeoff, num_doc_rows, z);
    TrainingTimeProfile::partial_z.Stop();
 
    // computes
    //    z = z + \ny y - I = H H^T \alpha - tradeoff \alpha + \nu y - I
    //    r_{dual} = ||\lambda - \xi + z||_2
    //    r_{pri} = |y^T \alpha|
    // here resd coresponds to r_{dual}, resp coresponds to r_{pri},
    // refer to formulae (8) and (11) in WG's paper.
    TrainingTimeProfile::check_stop.Start();
    resp = 0.0;
    resd = 0.0;
double *temp = new double[num_doc_rows];//register
    for (i = 0; i < num_doc_rows; ++i) {
      z[i] += nu * label[i] - 1.0;
      temp[i] = la[i] - xi[i] + z[i];
	}
	for (i = 0; i < num_doc_rows; ++i) {
      resd += temp[i] * temp[i];
	}
	for (i = 0; i < num_doc_rows; ++i) {
      resp += label[i] * x[i];
	}
	delete []temp;
    resp = fabs(resp);
    resd = sqrt(resd);
    if (parameter.verb >= 1) {
      cout << StringPrintf("r_pri: %-.10le r_dual: %-.10le\n",
                                resp,
                                resd);
    }
    // Converge Stop Condition. For more details refer to Algorithm 11.2
    // in Convex Optimization.
    if ((resp <= parameter.feas_thresh) &&
        (resd <= parameter.feas_thresh) &&
        (eta <= parameter.sgap)) {
      break;
    }
    TrainingTimeProfile::check_stop.Stop();
  
    // Update Variables
    //
    // computes
    //     tlx = \frac{1}{t \alpha}
    //     tux = \frac{1}{t (C - \alpha)}
    //     xilx = \frac{\xi}{\alpha}
    //     laux = \frac{\lambda}{C - \alpha}
    //     D^(-1) = \diag(\frac{\xi}{\alpha} + \frac{\lambda}{C - \alpha})
    // note D is a diagonal matrix and its inverse can be easily computed.
    TrainingTimeProfile::update_variables.Start();
    double m_lx, m_ux;
    for (i = 0; i < num_doc_rows; ++i) {
      m_lx = std::max(x[i], parameter.epsilon_x);
      m_ux = std::max(c[i] - x[i], parameter.epsilon_x);
      tlx[i] = 1.0 / (t * m_lx);
      tux[i] = 1.0 / (t * m_ux);
      xilx[i] = std::max(xi[i] / m_lx, parameter.epsilon_x);
      laux[i] = std::max(la[i] / m_ux, parameter.epsilon_x);
      d[i] = 1.0 / (xilx[i] + laux[i]);  // note here compute D^{-1} beforehand
	}
    // complete computation of z, note before
    // here z stores part of (17) except
    // the last term. Now complete z with
    // intermediates above, i.e. tlx and tux
    for (i = 0; i < num_doc_rows; ++i)
      z[i] = tlx[i] - tux[i] - z[i];
    TrainingTimeProfile::update_variables.Stop();
  
    // Newton Step
    // calculate icfA as E = I+H^T D H
    TrainingTimeProfile::production.Start();
#pragma offload target(mic)  in(d:length(row) alloc_if(0) free_if(0))
	{}
	  for (int i = 0; i < column; ++i) {
	    //offset += i;
	    #pragma offload target(mic) nocopy(icf_1) nocopy(d) nocopy(buff)
	    {
	 //ivdep --> ignores vector dependency.

	    #pragma ivdep
	   // #pragma omp parallel for schedule(runtime)
	    for (int p = 0; p < row; ++p) {
	      buff[p] = d[p] * icf_1[i* row + p];
	    }
	    }
	 
	     /*
	    Pulled the offload section out of outer for loop
	     */
	      #pragma offload target(mic) nocopy(icf_1) nocopy(buff) nocopy(icfa_1)
	      {
            #pragma omp parallel for schedule(runtime)
	    for (int j = 0; j <= i; ++j) {
	      double tmp = 0;
	      for (int p = 0; p < row; ++p) {
	        tmp += buff[p] * icf_1[j* row + p];
	      }
	      icfa_1[icfa_dim * j - j * (j + 1)/2 + i] = tmp;
	      }
	    }
	  }

#pragma offload target(mic) out(icfa_1:length(icfa_dim * (icfa_dim + 1) / 2) alloc_if(0) free_if(0))
{}
for(i = 0; i < column; ++i)
  icfa_1[icfa_dim * i - i * (i + 1)/2 + i] += 1;

    TrainingTimeProfile::production.Stop();

    // matrix cholesky factorization
    // note, rank is dimension of E, i.e.
    TrainingTimeProfile::cf.Start();
      MatrixManipulation_CF(icfa_1, lra, icfa_dim);
    TrainingTimeProfile::cf.Stop();
 
    // compute dnu = \Sigma^{-1}z, dx = \Sigma^{-1}(z - y \delta\nu), through
    // linear equations trick or Matrix Inversion Lemma
    TrainingTimeProfile::update_variables.Start();
    PrimalDualIPM_ComputeDeltaNu(rbicf, parallelmatrix, d, label, z, x, lra, icfa_dim, num_doc_rows, &dnu);//lra_dim = icfa_dim
    PrimalDualIPM_ComputeDeltaX(rbicf, parallelmatrix, d, label, dnu, lra, icfa_dim, z, num_doc_rows, dx);

    // update dxi, and dla
	//#pragma omp parallel for
    for (i = 0; i < num_doc_rows; ++i) {
      dxi[i] = tlx[i] - xilx[i] * dx[i] - xi[i];
      dla[i] = tux[i] + laux[i] * dx[i] - la[i];
    }

    // Line Search
    //
    // line search for primal and dual variable
    double ap = DBL_MAX;
    double ad = DBL_MAX;
    for (i = 0; i < num_doc_rows; ++i) {
      // make sure \alpha + \delta\alpha \in [\epsilon, C - \epsilon],
      // note here deal with positive and negative
      // search directionsituations seperately.
      // Refer to chapter 11 in Convex Optimization for more details.
      //c[i] = (label[i] > 0.0) ? c_pos : c_neg;
      if (dx[i]  > 0.0) {
        ap = std::min(ap, (c[i] - x[i]) / dx[i]);
      }
      if (dx[i]  < 0.0) {
        ap = std::min(ap, -x[i]/dx[i]);
      }
      // make sure \xi+ \delta\xi \in [\epsilon, +\inf), also
      // \lambda + \delta\lambda \in [\epsilon, +\inf).
      // deal with negative search direction.
      // Refer to chapter 11 in Convex Optimization for more details.
      if (dxi[i] < 0.0) {
        ad = std::min(ad, -xi[i] / dxi[i]);
      }
      if (dla[i] < 0.0) {
        ad = std::min(ad, -la[i] / dla[i]);
      }
	}

    // According to Primal-Dual IPM, the solution must be strictly feasible
    // to inequality constraints, here we add some disturbation to avoid
    // equality, for more details refer to 11.7.3 in Convex Optimization.
    //
    // Note according to 11.7.3 in Convex Optimization, here lack the
    // backsearch phase, but that's not the case, because of linear inequality
    // constraints, we always satisfy f(x^+) \preccurlyeq 0, refer to 11.7.3
    // in Convex Optimization.
    //
    ap = std::min(ap, 1.0) * 0.99;
    ad = std::min(ad, 1.0) * 0.99;

    // Update
    //
    // Update vectors \alpha, \xi, \lambda, and scalar \nu according to Newton
    // step and search direction. This completes one Newton's iteration, refer
    // to Algorithm 11.2 in Convex Optimization.
	//#pragma omp parallel for
    for (i = 0; i < num_doc_rows; ++i) {
      x[i]  += ap * dx[i];
      xi[i] += ad * dxi[i];
      la[i] += ad * dla[i];
    }
    nu += ad * dnu;
    TrainingTimeProfile::update_variables.Stop();
   
  }
//===================================for step end===============================

  // Not Convergent in specified iterations.
  // Note there are some other criteria of infeasibility.
  TrainingTimeProfile::ipm_misc.Start();
  if (step >= parameter.max_iter) {
    cout << StringPrintf("Maximum iterations (%d) has "
              "been reached before convergence,\n",
              parameter.max_iter);
    cout << StringPrintf("Please change the parameters.\n");
  }
  TrainingTimeProfile::ipm_misc.Stop();

  // write back the solutions
  TrainingTimeProfile::check_sv.Start();
  model->CheckSupportVector(x, doc, Sample_label,Sample_Feature_id,Sample_Feature_weight,Sample_two_norm_sq,parameter);
  TrainingTimeProfile::check_sv.Stop();

  // clean up
  TrainingTimeProfile::ipm_misc.Start();
delete []c;
  delete [] dx;
  delete [] x;
  delete [] xi;
  delete [] la;
  delete [] d;
  delete [] z;
  delete [] dxi;
  delete [] dla;
  delete [] tlx;
  delete [] tux;
  delete [] xilx;
  delete [] laux;
  delete [] label;
  TrainingTimeProfile::ipm_misc.Stop();
  return 0;
}

// Compute part of $z$, which is $H^TH\alpha$
int PrimalDualIPM_ComputePartialZ(   double ** icf,struct ParallelMatrix parallelmatrix,
                                      double *x,    double to,
                                      int num_doc_rows,
                                   double *z) {
 
  register int i, j;
  int p = parallelmatrix.num_cols_;
double *vz = new double[p];
  memset(vz, 0, sizeof(vz[0]) * p);
 
  double sum = 0.0;
  for (j = 0; j < p; ++j) {
    sum = 0.0;
  
    for (i = 0; i < num_doc_rows; ++i) {
   
      sum += icf[j][i] * x[i];
 
    }
  
    vz[j] = sum;
  }
  for (i = 0; i < num_doc_rows; ++i) {
    // Get a piece of inner product
    double sum2 = 0.0;
    for (j = 0; j < p; ++j) {
      sum2 += icf[j][i] * vz[j];
    }
    z[i] = sum2 - to * x[i];
  }

  delete [] vz;
  return 0;
}

// Compute surrogate gap
double PrimalDualIPM_ComputeSurrogateGap(double c_pos,
                                        double c_neg,
                                           double *label,
                                        int num_doc_rows,
                                           double *x,
                                           double *la,
                                           double *xi) {
  register int i;
register double sum = 0.0;
double *c = new double[num_doc_rows];
  for (i = 0; i < num_doc_rows; ++i) {
    c[i] = (label[i] > 0.0) ? c_pos : c_neg;
    sum += la[i] * c[i];
  }
  delete []c;
  for (i = 0; i < num_doc_rows; ++i) {
    sum += x[i] * (xi[i] - la[i]);
  }
  return sum;
}

// Compute Newton direction of primal variable $\alpha$
int PrimalDualIPM_ComputeDeltaX(   double ** icf,struct ParallelMatrix parallelmatrix,
                                    double *d,    double *label,
                                    double dnu,    double ** lra, int lra_dim,
                                    double *z, int num_doc_rows,
                                 double *dx) {
  register int i;
double *tz = new double[num_doc_rows];
//  #pragma omp parallel for 
  for (i = 0; i < num_doc_rows; ++i)
    tz[i] = z[i] - dnu * label[i];
  PrimalDualIPM_LinearSolveViaICFCol(icf, parallelmatrix, d, tz, lra, lra_dim, num_doc_rows, dx);
  // clean up
  delete [] tz;
  return 0;
}

// Compute Newton direction of primal variable $\nu$
int PrimalDualIPM_ComputeDeltaNu(   double ** icf,struct ParallelMatrix parallelmatrix,
                                     double *d,    double *label,
                                     double *z,    double *x,
                                     double ** lra, int lra_dim, int num_doc_rows,
                                  double *dnu) {
  register int i;
register double sum1 = 0.0;
register double sum2 = 0.0;
  // calculate inv(Q+D)*lz
double *tw = new double[num_doc_rows];
  PrimalDualIPM_LinearSolveViaICFCol(icf,parallelmatrix, d, z, lra, lra_dim, num_doc_rows, tw);
  // calculate inv(Q+D)*label
double *tl = new double[num_doc_rows];
double *l = new double[num_doc_rows];
  for (int i = 0; i < num_doc_rows; ++i)
    l[i] = label[i];
  PrimalDualIPM_LinearSolveViaICFCol(icf,parallelmatrix, d, l, lra, lra_dim, num_doc_rows, tl);
 // #pragma omp parallel for reduction(+:sum1)
  for (i = 0; i < num_doc_rows; ++i) {
    sum1 += label[i] * (tw[i] + x[i]);
  }
//#pragma omp parallel for reduction(+:sum2)
  for (i = 0; i < num_doc_rows; ++i) {
    sum2 += label[i] * tl[i];
  }
  // clean up
  delete [] tw;
  delete [] tl;
  delete [] l;
  *dnu = sum1 / sum2;
  return 0;
}

// solve a linear system via Sherman-Morrison-Woodbery formula
int PrimalDualIPM_LinearSolveViaICFCol(   double **  icf,struct ParallelMatrix parallelmatrix,
                                           double *d,
                                           double *b,
                                        double ** lra,int lra_dim,
                                        int num_doc_rows,
                                        double *x) {
  // Solve (D+HH')x = b using ICF and SMW update
  // V(dimxrank) : input matrix (smatrix)
  // D(dim)      : diagonal matrix in vector
  // b(dim)      : target vector
  // rank        : rank of ICF matrix
  register int i, j;
  int p = parallelmatrix.num_cols_;
double *vz = new double[p];
double *z  = new double[num_doc_rows];
  // we already inversed matrix before
  // calculate z=inv(D)*b[idx] 
 // #pragma omp parallel for
  for (i = 0; i < num_doc_rows; ++i)
    z[i] = b[i] * d[i];
  memset(vz, 0, sizeof(vz[0]) * p);
  double sum;
  for (j = 0; j < p; ++j) {
    sum = 0.0;
//#pragma omp parallel for reduction(+:sum)
    for (i = 0; i < num_doc_rows; ++i) {
      sum += icf[j][i] * z[i];
    }
    vz[j] = sum;
  }
double *ty = new double[p];
    MatrixManipulation_CholForwardSub(lra, vz, ty, lra_dim);
    MatrixManipulation_CholBackwardSub(lra, ty, vz, lra_dim);
    delete [] ty;
  // calculate u = z - inv(D)*V*t

  for (i = 0; i < num_doc_rows; ++i) {
    sum = 0.0;
//#pragma omp parallel for reduction(+:sum)
    for (j = 0; j < p; ++j) {
      sum += icf[j][i] * vz[j] * d[i];
    }
    x[i] = z[i] - sum;
  }
  // clean up
  delete [] z;
  delete [] vz;
  return 0;
}

// Loads the values of alpha, xi, lambda and nu to resume from an interrupted
// solving process.
void PrimalDualIPM_LoadVariables(
    const struct PrimalDualIPMParameter parameter,
    int num_total_doc, int *step,
    double *nu, double *x, double *la, double *xi) {
  char path[MAX_PATH_LEN];

  snprintf(path, sizeof(path), "%s/variables.saved_step", parameter.model_path);
  if (File::Exists(path)) {
    cout << "Intermedia Results found: " << path;
    Timer load_timer;
    load_timer.Start();
    int last_step = 0;
    File *file = File::OpenOrDie(path, "r");
    file->ReadOrDie(&last_step, sizeof(last_step));
    CHECK(file->Close());
    delete file;
    cout << "Resuming from step " << last_step << " ...";

    snprintf(path, sizeof(path), "%s/variables_step%05d",
             parameter.model_path, last_step);
    file = File::OpenOrDie(path, "r");

    int old_num_total_doc;
    CHECK(file->Read(step, sizeof(*step)) == sizeof(*step));
    CHECK(file->Read(&old_num_total_doc, sizeof(old_num_total_doc)) ==
          sizeof(old_num_total_doc));
    CHECK(old_num_total_doc == num_total_doc);

    CHECK(file->Read(nu, sizeof(*nu)) == sizeof(*nu));
    CHECK(file->Read(x, sizeof(x[0]) * num_total_doc) ==
          sizeof(x[0]) * num_total_doc);
    CHECK(file->Read(la, sizeof(la[0]) * num_total_doc) ==
          sizeof(la[0]) * num_total_doc);
    CHECK(file->Read(xi, sizeof(xi[0]) * num_total_doc) ==
          sizeof(xi[0]) * num_total_doc);
    CHECK(file->Close());
    delete file;
    load_timer.Stop();
    cout << "IPM resumed in " << load_timer.total() << " seconds" << endl;
  }
}

// Saves the values of alpha, xi, lambda and nu. num_local_doc, num_total_doc
// and num_processors are also saved to facilitate the loading procedure.
void PrimalDualIPM_SaveVariables(
    const struct PrimalDualIPMParameter parameter,
    int num_total_doc, int step,
    double nu, double *x, double *la, double *xi) {
  Timer save_timer;
  save_timer.Start();
  char path[MAX_PATH_LEN];
  int last_step = -1;
  File* file;

  snprintf(path, sizeof(path), "%s/variables.saved_step", parameter.model_path);
  if (File::Exists(path)) {
    file = File::OpenOrDie(path, "r");
    file->ReadOrDie(&last_step, sizeof(last_step));
    CHECK(file->Close());
    delete file;
  }
  if (step == last_step) return;

  cout << "Saving variables ... " << endl;
  snprintf(path, sizeof(path), "%s/variables_step%05d",
           parameter.model_path, step);
  file = File::OpenOrDie(path, "w");

  CHECK(file->Write(&step, sizeof(step)) == sizeof(step));
  CHECK(file->Write(&num_total_doc, sizeof(num_total_doc)) ==
        sizeof(num_total_doc));

  CHECK(file->Write(&nu, sizeof(nu)) == sizeof(nu));
  CHECK(file->Write(x, sizeof(x[0]) * num_total_doc) ==
        sizeof(x[0]) * num_total_doc);
  CHECK(file->Write(la, sizeof(la[0]) * num_total_doc) ==
        sizeof(la[0]) * num_total_doc);
  CHECK(file->Write(xi, sizeof(xi[0]) * num_total_doc) ==
        sizeof(xi[0]) * num_total_doc);
  CHECK(file->Flush());
  CHECK(file->Close());
  delete file;
    snprintf(path, sizeof(path), "%s/variables.saved_step",
             parameter.model_path);
    file = File::OpenOrDie(path, "w");
    file->WriteOrDie(&step, sizeof(step));
    CHECK(file->Flush());
    CHECK(file->Close());
    delete file;
  if (last_step != -1) {
    snprintf(path, sizeof(path), "%s/variables_step%05d",
             parameter.model_path, last_step);
    CHECK(file->Delete(path));
  }

  save_timer.Stop();
  cout << "Variables saved in " << save_timer.total()
            << " seconds" << endl;
}
}
