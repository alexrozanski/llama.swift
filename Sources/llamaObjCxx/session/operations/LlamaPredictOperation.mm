//
//  LlamaPredictOperation.m
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import "LlamaPredictOperation.hh"

#import "LlamaContext.hh"
#import "LlamaError.h"
#import "LlamaErrorInternal.h"
#import "LlamaPredictionEvent.h"
#import "LlamaSessionParams.h"

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

@interface LlamaPredictOperation () {
  LlamaContext *_context;
  NSString *_prompt;
  LlamaPredictOperationEventHandler _eventHandler;
  dispatch_queue_t _eventHandlerQueue;
}

@end

@implementation LlamaPredictOperation

- (instancetype)initWithIdentifier:(NSString *)identifier
                           context:(LlamaContext *)context
                            prompt:(NSString *)prompt
                      eventHandler:(LlamaPredictOperationEventHandler)eventHandler
                 eventHandlerQueue:(dispatch_queue_t)eventHandlerQueue
{
  if ((self = [super init])) {
    _identifier = [identifier copy];
    _context = context;
    _prompt = [prompt copy];
    _eventHandler = [eventHandler copy];
    _eventHandlerQueue = eventHandlerQueue;
  }

  return self;
}

- (void)main
{
  [self _postEvent:[_LlamaPredictionEvent started]];

  NSError *initializationError = nil;
  if (![self _initializeContextIfNeededWithError:&initializationError]) {
    [self _postEvent:[_LlamaPredictionEvent failedWithError:initializationError]];
    return;
  }

  if ([self _runPrediction]) {
    [self _postEvent:self.isCancelled ? [_LlamaPredictionEvent cancelled] : [_LlamaPredictionEvent completed]];
  }
}

- (BOOL)_runPrediction
{
  const int n_ctx = llama_n_ctx(_context.ctx);

  NSError *tokenizeError = nil;

  // prefix & suffix for instruct mode
  std::vector<llama_token> inp_pfx;
  if (![self _tokenizeString:"\n\n### Instruction:\n\n" into:inp_pfx addBeginningOfSequence:true outError:&tokenizeError]) {
    [self _postEvent:[_LlamaPredictionEvent failedWithError:tokenizeError]];
    return NO;
  }

  std::vector<llama_token> inp_sfx;
  if (![self _tokenizeString:"\n\n### Response:\n\n" into:inp_sfx addBeginningOfSequence:false outError:&tokenizeError]) {
    [self _postEvent:[_LlamaPredictionEvent failedWithError:tokenizeError]];
    return NO;
  }

  // determine newline token
  std::vector<llama_token> llama_token_newline;
  if (![self _tokenizeString:"\n" into:llama_token_newline addBeginningOfSequence:false outError:&tokenizeError]) {
    [self _postEvent:[_LlamaPredictionEvent failedWithError:tokenizeError]];
    return NO;
  }

  // run in interactive mode always so run the loop until we are finished.
  while (true) {
    if (self.isCancelled) {
      // even though we're cancelling this is still successful.
      return YES;
    }

    // predict
    if (_context.runState->embd.size() > 0) {
      // infinite text generation via context swapping
      // if we run out of context:
      // - take the n_keep first tokens from the original prompt (via n_past)
      // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
      if (_context.runState->n_past + (int)_context.runState->embd.size() > n_ctx) {
        const int n_left = _context.runState->n_past - _context.params.numberOfTokensToKeepFromInitialPrompt;

        _context.runState->n_past = _context.params.numberOfTokensToKeepFromInitialPrompt;

        // insert n_left/2 tokens at the start of embd from last_n_tokens
        _context.runState->embd.insert(_context.runState->embd.begin(), _context.runState->last_n_tokens.begin() + n_ctx - n_left / 2 - _context.runState->embd.size(), _context.runState->last_n_tokens.end() - _context.runState->embd.size());
      }

      if (llama_eval(_context.ctx, _context.runState->embd.data(), (int)_context.runState->embd.size(), _context.runState->n_past, _context.params.numberOfThreads)) {
        [self _postEvent:[_LlamaPredictionEvent failedWithError:makeLlamaError(LlamaErrorCodePredictionFailed, @"failed to eval")]];
        return NO;
      }
    }

    _context.runState->n_past += _context.runState->embd.size();
    _context.runState->embd.clear();

    if ((int)_context.runState->embd_inp.size() <= _context.runState->n_consumed) {
      // out of user input, sample next token
      const float top_k = _context.params.topK;
      const float top_p = _context.params.topP;
      const float temp = _context.params.temp;
      const float repeat_penalty = _context.params.repeatPenalty;

      llama_token id = 0;

      {
        auto logits = llama_get_logits(_context.ctx);

        id = llama_sample_top_p_top_k(_context.ctx,
                                      _context.runState->last_n_tokens.data() + n_ctx - _context.params.numberOfTokensToPenalize,
                                      _context.params.numberOfTokensToPenalize, top_k, top_p, temp, repeat_penalty);

        _context.runState->last_n_tokens.erase(_context.runState->last_n_tokens.begin());
        _context.runState->last_n_tokens.push_back(id);
      }

      // replace end of text token with newline token when in interactive mode
      if (id == llama_token_eos() && !_context.params.isInstructional) {
        id = llama_token_newline.front();
        NSString *firstAntiprompt = _context.params.antiprompts.firstObject;
        if (firstAntiprompt != nil) {
          // tokenize and inject first reverse prompt
          std::vector<llama_token> first_antiprompt;
          if (![self _tokenizeString:[firstAntiprompt cStringUsingEncoding:NSUTF8StringEncoding] into:first_antiprompt addBeginningOfSequence:false outError:&tokenizeError]) {
            [self _postEvent:[_LlamaPredictionEvent failedWithError:tokenizeError]];
            return NO;
          }
          _context.runState->embd_inp.insert(_context.runState->embd_inp.end(), first_antiprompt.begin(), first_antiprompt.end());
        }
      }

      // add it to the context
      _context.runState->embd.push_back(id);

      // decrement remaining sampling budget
      --_context.runState->n_remain;
    } else {
      // some user input remains from prompt or interaction, forward it to processing
      while ((int)_context.runState->embd_inp.size() > _context.runState->n_consumed) {
        _context.runState->embd.push_back(_context.runState->embd_inp[_context.runState->n_consumed]);
        _context.runState->last_n_tokens.erase(_context.runState->last_n_tokens.begin());
        _context.runState->last_n_tokens.push_back(_context.runState->embd_inp[_context.runState->n_consumed]);
        ++_context.runState->n_consumed;
        if ((int)_context.runState->embd.size() >= _context.params.batchSize) {
          break;
        }
      }
    }

    // return text results
    if (!self.isCancelled) {
      for (auto id : _context.runState->embd) {
        NSString *token = [NSString stringWithCString:llama_token_to_str(_context.ctx, id) encoding:NSUTF8StringEncoding];
        [self _postEvent:[_LlamaPredictionEvent outputTokenWithToken:token]];
      }
    }

    // if not currently processing queued inputs check if we should prompt the user for more
    if ((int)_context.runState->embd_inp.size() <= _context.runState->n_consumed) {
      // check for reverse prompt
      std::string last_output;
      for (auto id : _context.runState->last_n_tokens) {
        last_output += llama_token_to_str(_context.ctx, id);
      }

      // Check if each of the reverse prompts appears at the end of the output.
      for (NSString *antiprompt in _context.params.antiprompts) {
        const char *cString = [antiprompt cStringUsingEncoding:NSUTF8StringEncoding];
        NSUInteger length = [antiprompt lengthOfBytesUsingEncoding:NSUTF8StringEncoding];
        if (last_output.find(cString, last_output.length() - length, length) != std::string::npos) {
          return YES;
        }
      }
    }

    // end of text token
    if (_context.runState->embd.back() == llama_token_eos()) {
      return YES;
    }

    // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
    if (_context.runState->n_remain <= 0 && _context.params.numberOfTokens != -1) {
      _context.runState->n_remain = _context.params.numberOfTokens;
      return YES;
    }
  }

  // should be a noop
  return YES;
}

#pragma mark - Private

- (BOOL)_initializeContextIfNeededWithError:(NSError **)outError
{
  return [_context initializeWithInitializationBlock:^(LlamaContext *context, NSError **outError) {
    std::string prompt([self->_prompt cStringUsingEncoding:NSUTF8StringEncoding]);

    // Add a space in front of the first character to match OG llama tokenizer behavior
    prompt.insert(0, 1, ' ');

    // Initialize the run state.
    const int n_ctx = llama_n_ctx(context.ctx);
    auto runState = context.runState;

    runState->n_past = 0;
    runState->n_remain = context.params.numberOfTokens;
    runState->n_consumed = 0;

    runState->last_n_tokens.resize(n_ctx);
    runState->last_n_tokens.assign(n_ctx, 0);

    // tokenize the initial prompt
    if (![self _tokenizeString:prompt into:context.runState->embd_inp addBeginningOfSequence:true outError:outError]) {
      return NO;
    }

    // Remaining setup.
    if ((int)context.runState->embd_inp.size() > n_ctx - 4) {
      if (outError) {
        *outError = makeLlamaError(LlamaErrorCodePredictionFailed, [NSString stringWithFormat:@"prompt is too long (%d tokens, max %d)\n", (int)_context.runState->embd_inp.size(), n_ctx - 4]);
      }
      return NO;
    }

    context.params.numberOfTokensToKeepFromInitialPrompt = std::min(context.params.numberOfTokensToKeepFromInitialPrompt, (int)context.runState->embd_inp.size());

    return YES;
  } outError: outError];
}

// adapted from llama_tokenize() from common.h
- (BOOL)_tokenizeString:(const std::string &)string
                   into:(std::vector<llama_token> &)tokens
 addBeginningOfSequence:(bool)addBeginningOfSequence
               outError:(NSError **)outError
{
  // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
  std::vector<llama_token> res(string.size() + (int)addBeginningOfSequence);
  int n = llama_tokenize(_context.ctx, string.c_str(), res.data(), (int)res.size(), addBeginningOfSequence, outError);
  if (n < 0) {
    return NO;
  }
  res.resize(n);
  tokens = res;
  return YES;
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
