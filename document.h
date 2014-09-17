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
#ifndef DOCUMENT_H__
#define DOCUMENT_H__

#include <vector>

using namespace std;
namespace psvm {

  // Reads samples from the file specified by filename. If the file does not
  // exist or the file format is illegal, false is returned. Otherwise true
  // is returned. The file format whould strickly be:
  //    label word-id:word-weight word-id:word-weight ...
  //    label word-id:word-weight word-id:word-weight ...
  bool Document_Read(const char* filename,struct Document doc, int *Sample_lable, double *Sample_two_norm_sq, int *Sample_Feature_id, double *Sample_Feature_weight);
  bool Document_Count(const char* filename,struct Document *doc);

struct Document{
  int num_total_;
  int num_pos_;
  int num_neg_;
  int num_feature;
};

}
#endif
