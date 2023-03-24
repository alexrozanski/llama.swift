//
//  LlamaPredictOperation.m
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import "LlamaPredictOperation.hh"

#import "LlamaError.h"
#import "LlamaEvent.h"
#import "LlamaSessionConfig.h"

#include "ggml.h"

#include "utils.hh"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#endif

@interface LlamaPredictOperation () {
  gpt_params _params;
  LlamaPredictOperationEventHandler _eventHandler;
  dispatch_queue_t _eventHandlerQueue;

  bool _isInteracting;
}

@end

@implementation LlamaPredictOperation

- (instancetype)initWithParams:(gpt_params)params
                  eventHandler:(LlamaPredictOperationEventHandler)eventHandler
             eventHandlerQueue:(dispatch_queue_t)eventHandlerQueue
{
  if ((self = [super init])) {
    _params = params;
    _eventHandler = [eventHandler copy];
    _eventHandlerQueue = eventHandlerQueue;
  }

  return self;
}

- (void)main
{
  // has to be called once at the start of the program to init ggml stuff
  ggml_time_init();

  gpt_params params(_params);
//  params.model = "models/llama-7B/ggml-model.bin";

//  if (gpt_params_parse(argc, argv, params) == false) {
//    return 1;
//  }

  if (_params.n_ctx > 2048) {
    NSLog(@"warning: model does not support context sizes greater than 2048 tokens (%d specified);"
          "expect poor results\n", _params.n_ctx);
  }

  if (_params.seed <= 0) {
    _params.seed = time(NULL);
  }

  std::mt19937 rng(_params.seed);
  if (_params.random_prompt) {
    _params.prompt = gpt_random_prompt(rng);
  }

  // save choice to use color for later
  // (note for later: this is a slightly awkward choice)
//      con_use_color = params.use_color;

  llama_context * ctx;

  // load the model
  {
    [self postEvent:[_LlamaEvent startedLoadingModel]];

    auto lparams = llama_context_default_params();

    lparams.n_ctx      = _params.n_ctx;
    lparams.n_parts    = _params.n_parts;
    lparams.seed       = _params.seed;
    lparams.f16_kv     = _params.memory_f16;
    lparams.logits_all = _params.perplexity;

    NSError *loadError = nil;
    ctx = llama_init_from_file(_params.model.c_str(), lparams, &loadError);

    if (ctx == NULL) {
      [self postEvent:[_LlamaEvent failedWithError:loadError]];
      return;
    }
  }

  // print system information
//  {
//    fprintf(stderr, "\n");
//    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
//            params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
//  }

  [self postEvent:[_LlamaEvent startedGeneratingOutput]];

  // determine the required inference memory per token:
  // TODO: better way to do that
  {
    const std::vector<llama_token> tmp = { 0, 1, 2, 3 };
    NSError *evalError = nil;
    if (llama_eval(ctx, tmp.data(), tmp.size(), 0, _params.n_threads, &evalError) != 0) {
      return;
    }
  }

//  if (params.perplexity) {
//    perplexity(ctx, params);
//    exit(0);
//  }

  int n_past = 0;

  // Add a space in front of the first character to match OG llama tokenizer behavior
  _params.prompt.insert(0, 1, ' ');

  // tokenize the prompt
  bool tokenizeSuccess = false;
  NSError *tokenizeError = nil;
  auto embd_inp = ::llama_tokenize(ctx, _params.prompt, true, &tokenizeSuccess, &tokenizeError);
  if (!tokenizeSuccess) {
    return;
  }

  const int n_ctx = llama_n_ctx(ctx);

  _params.n_predict = std::min(_params.n_predict, n_ctx - (int) embd_inp.size());

  // prefix & suffix for instruct mode
  const auto inp_pfx = ::llama_tokenize(ctx, "\n\n### Instruction:\n\n", true, &tokenizeSuccess, &tokenizeError);
  if (!tokenizeSuccess) {
    return;
  }

  const auto inp_sfx = ::llama_tokenize(ctx, "\n\n### Response:\n\n", false,  &tokenizeSuccess, &tokenizeError);
  if (!tokenizeSuccess) {
    return;
  }

  // in instruct mode, we inject a prefix and a suffix to each input by the user
  if (_params.instruct) {
    _params.interactive = true;
    _params.antiprompt.push_back("### Instruction:\n\n");
  }

  // enable interactive mode if reverse prompt is specified
  if (_params.antiprompt.size() != 0) {
    _params.interactive = true;
  }

  //  fprintf(stderr, "\n");
  //  fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
  //  fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
  //  for (int i = 0; i < (int) embd_inp.size(); i++) {
  //      fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
  //  }
  //  fprintf(stderr, "\n");
  //  if (params.interactive) {
  //#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
  //      struct sigaction sigint_action;
  //      sigint_action.sa_handler = sigint_handler;
  //      sigemptyset (&sigint_action.sa_mask);
  //      sigint_action.sa_flags = 0;
  //      sigaction(SIGINT, &sigint_action, NULL);
  //#elif defined (_WIN32)
  //      signal(SIGINT, sigint_handler);
  //#endif
  //
  //      fprintf(stderr, "%s: interactive mode on.\n", __func__);
  //
  //      if(params.antiprompt.size()) {
  //          for (auto antiprompt : params.antiprompt) {
  //              fprintf(stderr, "Reverse prompt: '%s'\n", antiprompt.c_str());
  //          }
  //      }
  //  }
  //  fprintf(stderr, "sampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
  //  fprintf(stderr, "\n\n");

  std::vector<llama_token> embd;

  int last_n_size = params.repeat_last_n;
  std::vector<llama_token> last_n_tokens(last_n_size);
  std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

  //      if (params.interactive) {
  //          fprintf(stderr, "== Running in interactive mode. ==\n"
  //  #if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
  //                 " - Press Ctrl+C to interject at any time.\n"
  //  #endif
  //                 " - Press Return to return control to LLaMa.\n"
  //                 " - If you want to submit another line, end your input in '\\'.\n\n");
  //          is_interacting = true;
  //      }

  int input_consumed = 0;
  bool input_noecho = false;

  int remaining_tokens = _params.n_predict;

//#if defined (_WIN32)
//  if (params.use_color) {
//        // Enable ANSI colors on Windows 10+
//        unsigned long dwMode = 0;
//        void* hConOut = GetStdHandle((unsigned long)-11); // STD_OUTPUT_HANDLE (-11)
//        if (hConOut && hConOut != (void*)-1 && GetConsoleMode(hConOut, &dwMode) && !(dwMode & 0x4)) {
//            SetConsoleMode(hConOut, dwMode | 0x4); // ENABLE_VIRTUAL_TERMINAL_PROCESSING (0x4)
//        }
//    }
//#endif
    // the first thing we will do is to output the prompt, so set color accordingly
//    set_console_state(CONSOLE_STATE_PROMPT);

  while (remaining_tokens > 0 || params.interactive) {
    // predict
    if (embd.size() > 0) {
      NSError *predictError = nil;
      if (llama_eval(ctx, embd.data(), embd.size(), n_past, params.n_threads, &predictError) != 0) {
        return;
      }
    }

    n_past += embd.size();
    embd.clear();

    if ((int) embd_inp.size() <= input_consumed) {
      // out of user input, sample next token
      const float top_k          = params.top_k;
      const float top_p          = params.top_p;
      const float temp           = params.temp;
      const float repeat_penalty = params.repeat_penalty;

      llama_token id = 0;

      {
        auto logits = llama_get_logits(ctx);

        if (params.ignore_eos) {
          // set the logit of the eos token to zero to avoid sampling it
          //logits[logits.size() - n_vocab + EOS_TOKEN_ID] = 0;
          // TODO: this does not work of params.logits_all == true
          assert(params.perplexity == false);
          logits[llama_token_eos()] = 0;
        }

        id = llama_sample_top_p_top_k(ctx, last_n_tokens.data(), last_n_tokens.size(), top_k, top_p, temp, repeat_penalty);

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
      }

      // add it to the context
      embd.push_back(id);

      // echo this to console
      input_noecho = false;

      // decrement remaining sampling budget
      --remaining_tokens;
    } else {
      // some user input remains from prompt or interaction, forward it to processing
      while ((int) embd_inp.size() > input_consumed) {
        embd.push_back(embd_inp[input_consumed]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(embd_inp[input_consumed]);
        ++input_consumed;
        if ((int) embd.size() >= params.n_batch) {
          break;
        }
      }
    }

    // display text
    if (!input_noecho) {
      for (auto id : embd) {
        printf("%s", llama_token_to_str(ctx, id));
      }
      fflush(stdout);
    }
    // reset color to default if we there is no pending user input
//    if (!input_noecho && (int)embd_inp.size() == input_consumed) {
//      set_console_state(CONSOLE_STATE_DEFAULT);
//    }

    // in interactive mode, and not currently processing queued inputs;
    // check if we should prompt the user for more
    if (params.interactive && (int) embd_inp.size() <= input_consumed) {
      // check for reverse prompt
      std::string last_output;
      for (auto id : last_n_tokens) {
        last_output += llama_token_to_str(ctx, id);
      }

      // Check if each of the reverse prompts appears at the end of the output.
      for (std::string antiprompt : params.antiprompt) {
        if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
          _isInteracting = true;
          break;
        }
      }
      if (_isInteracting) {
        // potentially set color to indicate we are taking user input
//        set_console_state(CONSOLE_STATE_USER_INPUT);

        if (params.instruct) {
          input_consumed = embd_inp.size();
          embd_inp.insert(embd_inp.end(), inp_pfx.begin(), inp_pfx.end());

          printf("\n> ");
        }

        std::string buffer;
        std::string line;
        bool another_line = true;
        do {
          std::getline(std::cin, line);
          if (line.empty() || line.back() != '\\') {
            another_line = false;
          } else {
            line.pop_back(); // Remove the continue character
          }
          buffer += line + '\n'; // Append the line to the result
        } while (another_line);

        // done taking input, reset color
//        set_console_state(CONSOLE_STATE_DEFAULT);

        auto line_inp = ::llama_tokenize(ctx, buffer, false, &tokenizeSuccess, &tokenizeError);
        if (!tokenizeSuccess) {
          return;
        }
        embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

        if (params.instruct) {
          embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
        }

        remaining_tokens -= line_inp.size();

        input_noecho = true; // do not echo this again
      }
      _isInteracting = false;
    }

    // end of text token
    if (embd.back() == llama_token_eos()) {
      if (params.interactive) {
        _isInteracting = true;
      } else {
        fprintf(stderr, " [end of text]\n");
        break;
      }
    }

    // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
    if (params.interactive && remaining_tokens <= 0) {
      remaining_tokens = params.n_predict;
      _isInteracting = true;
    }
  }

//#if defined (_WIN32)
//    signal(SIGINT, SIG_DFL);
//#endif

//  llama_print_timings(ctx);

  llama_free(ctx);

//  set_console_state(CONSOLE_STATE_DEFAULT);
}

- (void)postEvent:(_LlamaEvent *)event
{
  dispatch_async(_eventHandlerQueue, ^() {
    if (self->_eventHandler != NULL) {
      self->_eventHandler(event);
    }
  });
}

@end
