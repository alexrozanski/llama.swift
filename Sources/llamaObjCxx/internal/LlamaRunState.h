//
//  LlamaRunState.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 28/03/2023.
//

#include <stdio.h>
#include <string>
#include <vector>

typedef struct {
  std::string path_session;
  std::vector<llama_token> session_tokens;

  std::vector<llama_token> embd;
  int n_past;
  int n_remain;
  int n_consumed;
  int n_session_consumed;

  std::vector<llama_token> embd_inp;

  // TODO: replace with ring-buffer when done so in llama.cpp
  std::vector<llama_token> last_n_tokens;

  bool is_antiprompt;
  bool need_to_save_session;
} llama_swift_run_state;
