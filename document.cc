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
#include <vector>
#include <string>
#include <utility>
#include <iostream>

#include "document.h"
#include "util.h"
#include "io.h"

namespace psvm {
bool Document_Count(const char* filename,struct Document *doc) {
  doc->num_total_ = 0;
  doc->num_pos_   = 0;
  doc->num_neg_   = 0;
  doc->num_feature = 0;
  int num_feature_tmp = 0;
  if (filename == NULL) return false;

  // Walks through each sample
  File* file = File::Open(filename, "r");
  if (file == NULL) {
    cerr << "Cannot find file " << filename << endl;
    return false;
  }
  string line;

  while (file->ReadLine(&line)) {
      int label = 0;
      const char* start = line.c_str();
      // Extracts the sample's class label
      if (!SplitOneIntToken(&start, " ", &label)) {
        cerr << "Error parsing line: " << doc->num_total_ + 1 << endl;
        return false;
      }

      // Gets the number of positive and negative samples
      if (label == 1) {
		  ++doc->num_pos_;
      } else if (label == -1) {
		  ++doc->num_neg_;
      } else {
        cerr << "Unknow label in line: " << doc->num_total_ + 1 << label;
        return false;
      }

      // Extracts the sample's features
      num_feature_tmp = 0;
      vector<pair<string, string> > kv_pairs;
      SplitStringIntoKeyValuePairs(string(start), ":", " ", &kv_pairs);
      vector<pair<string, string> >::const_iterator pair_iter;
      for (pair_iter = kv_pairs.begin(); pair_iter != kv_pairs.end();
           ++pair_iter) {
      num_feature_tmp ++;
	}
      if(doc->num_feature < num_feature_tmp)
          doc->num_feature = num_feature_tmp;
    ++doc->num_total_;
  }
  file->Close();
  delete file;
  return true;
}

bool Document_Read(const char* filename,struct Document doc, int *Sample_lable, double *Sample_two_norm_sq, int *Sample_Feature_id, double *Sample_Feature_weight) {
int num_total_count = 0;
int num_feature_count = 0;
double two_norm_sq_tmp = 0.0;
  if (filename == NULL) return false;
  // Walks through each sample
  File* file = File::Open(filename, "r");
  if (file == NULL) {
    cerr << "Cannot find file " << filename << endl;
    return false;
  }
  string line;
  while (file->ReadLine(&line)) {
      int label = 0;
      const char* start = line.c_str();
      if (!SplitOneIntToken(&start, " ", &label)) {
        cerr << "Error parsing line: " << doc.num_total_ + 1 << endl;
        return false;
      }
      Sample_lable[num_total_count] = label;
      two_norm_sq_tmp = 0.0;
      
      num_feature_count = 0;
      vector<pair<string, string> > kv_pairs;
      SplitStringIntoKeyValuePairs(string(start), ":", " ", &kv_pairs);
      vector<pair<string, string> >::const_iterator pair_iter;
      for (pair_iter = kv_pairs.begin(); pair_iter != kv_pairs.end();
           ++pair_iter) {
        Sample_Feature_id[num_total_count * doc.num_feature + num_feature_count] = atoi(pair_iter->first.c_str());
        Sample_Feature_weight[num_total_count * doc.num_feature + num_feature_count] = atof(pair_iter->second.c_str());
        two_norm_sq_tmp += Sample_Feature_weight[num_total_count * doc.num_feature + num_feature_count] * Sample_Feature_weight[num_total_count * doc.num_feature + num_feature_count];
        ++num_feature_count;
      }
      Sample_two_norm_sq[num_total_count] = two_norm_sq_tmp;
    ++num_total_count;
  }
  file->Close();
  delete file;
  return true;
}

}
