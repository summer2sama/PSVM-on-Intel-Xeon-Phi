PSVM-on-Intel-Xeon-Phi
======================
This is a quick start manual for PSVM-on-Intel-Xeon-Phi.

Introduction
=============
- PSVM-on-Intel-Xeon-Phi is an application for PSVM algorithm. It is supposed to run on CPU with Intel Xeon Phi as co-processors.
- The algorithm PSVM is from the following paper: http://books.nips.cc/papers/files/nips20/NIPS2007_0435.pdf. It is an all-kernel-support version of SVM, which can parallel run on multiple machines. PSVM Open Source is available for download at http://code.google.com/p/psvm/.
- This code is based on the open source code mentioned above.
- PSVM-on-Intel-Xeon-Phi must be run in linux environment. And for hardware configuration, it require Intel Xeon Phi chips.

Prepare datafile
=================
- Required datafile is similar to libsvm. Data is stored using a sparse representation, with one element per line. Each line begins with an integer, which is the element's label and which must be either -1 or 1. The label is then followed by a list of features, of the form featureID:featureValue. It is like <label> <index1>:<value1> <index2>:<value2> ... We only support binary classification, so the label must be 1/-1. Feature index must be in increasing order. 
- Example: Suppose there are two elements, each with two features. The first one is Feature0: 1, Feature1: 2 and belongs to class 1; The second one is Feature0: 100, Feature1: 200 and belongs to class -1. Then the datafile would look like:

  1  0:1    1:2
  
  -1 0:100  1:200

Compile
========
- We use icpc compiler here. 
- To compare the performance between circumstances with and withour using Intel Xeon Phi as co-processor, we could compile 2 version:
    - ``$ icpc matrix.cc model.cc timer.cc document.cc matrix_manipulation.cc pd_ipm.cc io.cc kernel.cc svm_train.cc util.cc -no-offload -o train_no_offload_baseline -O3 -vec-report2``
    - ``$ icpc matrix.cc model.cc timer.cc document.cc matrix_manipulation.cc pd_ipm.cc io.cc kernel.cc svm_train.cc util.cc -openmp -o train_offload_openmp -O3 -vec-report2``

Train
========
- Dowload a6a from http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a6a as training datafile.
- Train
  - ``$ mkdir /home/$USER/PSVM-on-Intel-Xeon-Phi/model/``
  - ``$ ./train_no_offload_baseline -rank_ratio 0.1 -kernel_type 2 -hyper_parm 1 -gamma 0.01 -model_path /home/zetta/wuyang/PSVM-on-Intel-Xeon-Phi/model a6a``
  - ``$ ./train_offload_openmp -rank_ratio 0.1 -kernel_type 2 -hyper_parm 1 -gamma 0.01 -model_path /home/zetta/wuyang/PSVM-on-Intel-Xeon-Phi/model a6a``
- You will see files about model, time-consumption in the /home/$USER/PSVM-on-Intel-Xeon-Phi/model/ folder after a training.

Command-line flags
==================
- kernel_type: 
  - 0: normalized-linear 
  - 1: normalized-polynomial 
  - 2: RBF Gaussian 
  - 3: Laplasian 
- rank_ratio: approximation ratio between 0 and 1. Higher values yield higher accuracy at the cost of increased memory usage and training time. If you are unsure what value to use, we recommend a value of 1/sqrt(n), where n is the number of training samples. 
- hyper_parm: C in SVM. This is the same as libsvm "-c" parameter 
- gamma: gamma value if you use RBF kernel. This is the same as libsvm "-g" parameter 
- poly_degree: degree if you use normalized-polynomial kernel. This is the same as libsvm "-d" parameter 
- model_path: the location to save the training model and checkpoints to. Be sure that this path is EMPTY before training a new model: svm_train will interpret any of its checkpoints left in this directory as checkpoints for the current model.
- failsafe: If failsafe is set to true, program will periodically write checkpoints to model_path and if program fail, it will restart from last checkpoints. 
- save_interval: Because PSVM supports failsafe. On every save_interval seconds, program will write a checkpoint. If PSVM fails such as machine is down, it will restart from last checkpoint on next execution. 
- surrogate_gap_threshold, feasible_threshold, max_iteration: Because PSVM use Interior Point Method, there needs many iterations. The iteration will stop by checking ((surrogate_gap < surrogate_gap_threshold and primal residual < feasible_threshold and dual residual < feasible_threshold) or iterations > max_iteration). Usually setting them to default will handle most of the cases. 
- positive_weight, negative_weight: For unbalanced data, we should set a more-than-one weight to one of the class. For example there are 100 positive data and 10 negative data, it is suggested you set negative_weight to 10. 
- Others: simply run svm_train to get description for each parameter. They are not frequently used unless you are quite familiar with algorithm details. 
