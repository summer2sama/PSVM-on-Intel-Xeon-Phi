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

#ifndef SVM_TRAIN_H__
#define SVM_TRAIN_H__

#include <string>

namespace psvm {
struct Document;
struct Kernel;
class Model;
struct PrimalDualIPMParameter;

class SvmTrainer {
 public:
  void TrainModel(   struct Document doc,int *Sample_Feature_id, double *Sample_Feature_weight,
                            double *Sample_two_norm_sq, int *Sample_label,
                               struct Kernel kernel,
                               struct PrimalDualIPMParameter parameter,
                            Model* model, bool failsafe);

  // Format the training time info of current processor to a string.
  std::string PrintTimeInfo();

  // Save time statistic information into a file.
  void SaveTimeInfo(const char *path, const char* file_name);
};
}
#endif
