//
//  llama_swift_run_state.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 28/03/2023.
//

#include <stdio.h>
#include <vector>

typedef struct {
  std::vector<llama_token> embd;
  int n_past;
  int n_remain;
  int n_consumed;
} llama_swift_run_state;
