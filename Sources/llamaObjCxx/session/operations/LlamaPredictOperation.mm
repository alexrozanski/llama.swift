//
//  LlamaPredictOperation.m
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import "LlamaPredictOperation.hh"

#import "LlamaContext.hh"
#import "LlamaError.h"
#import "LlamaPredictionEvent.h"
#import "LlamaSessionConfig.h"

#include "ggml.h"

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
  LlamaContext *_context;
  LlamaPredictOperationEventHandler _eventHandler;
  dispatch_queue_t _eventHandlerQueue;

  bool _isInteracting;
}

@end

@implementation LlamaPredictOperation

- (instancetype)initWithContext:(LlamaContext *)context
                         params:(gpt_params)params
                   eventHandler:(LlamaPredictOperationEventHandler)eventHandler
              eventHandlerQueue:(dispatch_queue_t)eventHandlerQueue;
{
  if ((self = [super init])) {
    _context = context;
    _params = params;
    _eventHandler = [eventHandler copy];
    _eventHandlerQueue = eventHandlerQueue;
  }

  return self;
}

- (void)main
{
  [self _postEvent:[_LlamaPredictionEvent started]];

  int n_past = 0;

  // Add a space in front of the first character to match OG llama tokenizer behavior
  _params.prompt.insert(0, 1, ' ');

  // tokenize the prompt
  bool tokenizeSuccess = false;
  NSError *tokenizeError = nil;
  auto embd_inp = ::llama_tokenize(_context.ctx, _params.prompt, true, &tokenizeSuccess, &tokenizeError);
  if (!tokenizeSuccess) {
    [self _postEvent:[_LlamaPredictionEvent failedWithError:tokenizeError]];
    return;
  }

  const int n_ctx = llama_n_ctx(_context.ctx);

  _params.n_predict = std::min(_params.n_predict, n_ctx - (int) embd_inp.size());

  // prefix & suffix for instruct mode
  const auto inp_pfx = ::llama_tokenize(_context.ctx, "\n\n### Instruction:\n\n", true, &tokenizeSuccess, &tokenizeError);
  if (!tokenizeSuccess) {
    [self _postEvent:[_LlamaPredictionEvent failedWithError:tokenizeError]];
    return;
  }

  const auto inp_sfx = ::llama_tokenize(_context.ctx, "\n\n### Response:\n\n", false,  &tokenizeSuccess, &tokenizeError);
  if (!tokenizeSuccess) {
    [self _postEvent:[_LlamaPredictionEvent failedWithError:tokenizeError]];
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

  std::vector<llama_token> embd;

  int last_n_size = _params.repeat_last_n;
  std::vector<llama_token> last_n_tokens(last_n_size);
  std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

  int input_consumed = 0;
  bool input_noecho = false;

  int remaining_tokens = _params.n_predict;

  while (remaining_tokens > 0 || _params.interactive) {
    // predict
    if (embd.size() > 0) {
      NSError *predictError = nil;
      if (llama_eval(_context.ctx, embd.data(), embd.size(), n_past, _params.n_threads, &predictError) != 0) {
        [self _postEvent:[_LlamaPredictionEvent failedWithError:predictError]];
        return;
      }
    }

    n_past += embd.size();
    embd.clear();

    if ((int) embd_inp.size() <= input_consumed) {
      // out of user input, sample next token
      const float top_k          = _params.top_k;
      const float top_p          = _params.top_p;
      const float temp           = _params.temp;
      const float repeat_penalty = _params.repeat_penalty;

      llama_token id = 0;

      {
        auto logits = llama_get_logits(_context.ctx);

        if (_params.ignore_eos) {
          // set the logit of the eos token to zero to avoid sampling it
          //logits[logits.size() - n_vocab + EOS_TOKEN_ID] = 0;
          // TODO: this does not work of params.logits_all == true
          assert(_params.perplexity == false);
          logits[llama_token_eos()] = 0;
        }

        id = llama_sample_top_p_top_k(_context.ctx, last_n_tokens.data(), last_n_tokens.size(), top_k, top_p, temp, repeat_penalty);

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
        if ((int) embd.size() >= _params.n_batch) {
          break;
        }
      }
    }

    // display text
    if (!input_noecho) {
      for (auto id : embd) {
        NSString *token = [NSString stringWithCString:llama_token_to_str(_context.ctx, id) encoding:NSUTF8StringEncoding];
        [self _postEvent:[_LlamaPredictionEvent outputTokenWithToken:token]];
      }
    }

    // in interactive mode, and not currently processing queued inputs;
    // check if we should prompt the user for more
    if (_params.interactive && (int) embd_inp.size() <= input_consumed) {
      // check for reverse prompt
      std::string last_output;
      for (auto id : last_n_tokens) {
        last_output += llama_token_to_str(_context.ctx, id);
      }

      // Check if each of the reverse prompts appears at the end of the output.
      for (std::string antiprompt : _params.antiprompt) {
        if (last_output.find(antiprompt.c_str(), last_output.length() - antiprompt.length(), antiprompt.length()) != std::string::npos) {
          _isInteracting = true;
          break;
        }
      }
      if (_isInteracting) {
        if (_params.instruct) {
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

        auto line_inp = ::llama_tokenize(_context.ctx, buffer, false, &tokenizeSuccess, &tokenizeError);
        if (!tokenizeSuccess) {
          [self _postEvent:[_LlamaPredictionEvent failedWithError:tokenizeError]];
          return;
        }
        embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

        if (_params.instruct) {
          embd_inp.insert(embd_inp.end(), inp_sfx.begin(), inp_sfx.end());
        }

        remaining_tokens -= line_inp.size();

        input_noecho = true; // do not echo this again
      }
      _isInteracting = false;
    }

    // end of text token
    if (embd.back() == llama_token_eos()) {
      if (_params.interactive) {
        _isInteracting = true;
      } else {
        fprintf(stderr, " [end of text]\n");
        break;
      }
    }

    // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
    if (_params.interactive && remaining_tokens <= 0) {
      remaining_tokens = _params.n_predict;
      _isInteracting = true;
    }
  }

  [self _postEvent:[_LlamaPredictionEvent completed]];
}

- (void)_postEvent:(_LlamaPredictionEvent *)event
{
  dispatch_async(_eventHandlerQueue, ^() {
    if (self->_eventHandler != NULL) {
      self->_eventHandler(event);
    }
  });
}

@end
