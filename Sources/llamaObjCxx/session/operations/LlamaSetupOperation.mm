//
//  LlamaSetupOperation.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 23/03/2023.
//

#import "LlamaSetupOperation.hh"

#import "LlamaContext.hh"
#import "LlamaError.h"
#import "LlamaErrorInternal.h"
#import "LlamaEvent.h"
#import "LlamaOperationUtils.hh"
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

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#endif

@interface LlamaSetupOperation () {
  _LlamaSessionParams *_params;
  LlamaSetupOperationEventHandler _eventHandler;
  dispatch_queue_t _eventHandlerQueue;
}

@end

@implementation LlamaSetupOperation

@synthesize delegate = _delegate;

- (instancetype)initWithParams:(_LlamaSessionParams *)params delegate:(id<LlamaSetupOperationDelegate>)delegate
{
  if ((self = [super init])) {
    _params = params;
    _delegate = delegate;
  }

  return self;
}

- (void)main
{
  LlamaContext *context = nil;
  NSError *setUpError = nil;

  if (![self _setUpReturningContext:&context error:&setUpError]) {
    dispatch_async(dispatch_get_main_queue(), ^{
      [self->_delegate setupOperation:self didFailWithError:setUpError];
    });
  } else {
    dispatch_async(dispatch_get_main_queue(), ^{
      [self->_delegate setupOperation:self didSucceedWithContext:context];
    });
  }
}

- (BOOL)_setUpReturningContext:(LlamaContext **)outContext error:(NSError **)outError
{
  if (_params.contextSize > 2048) {
    NSLog(@"warning: model does not support context sizes greater than 2048 tokens (%d specified);"
          "expect poor results", _params.contextSize);
  } else if (_params.contextSize < 8) {
    NSLog(@"warning: minimum context size is 8, using minimum size.\n");
    _params.contextSize = 8;
  }

  if (_params.seed < 0) {
    _params.seed = time(NULL);
  }

  llama_init_backend();

  llama_context * ctx;

  // load the model
  {
    auto lparams = llama_context_default_params();

    lparams.n_ctx        = _params.contextSize;
    lparams.n_batch      = _params.batchSize;
//    lparams.n_gpu_layers = _params.numberOfGPULayers;
//    lparams.main_gpu     = _params.main_gpu;
//    lparams.low_vram     = _params.low_vram;
    lparams.seed         = _params.seed;
    lparams.f16_kv       = _params.useF16Memory;
    lparams.use_mmap     = _params.useMmap;
    lparams.use_mlock    = _params.useMlock;

    const char *modelPath = [_params.modelPath cStringUsingEncoding:NSUTF8StringEncoding];
    ctx = llama_init_from_file(modelPath, lparams, outError);
    if (ctx == NULL) {
      return NO;
    }
  }

  if (_params.loraAdapter.length > 0) {
    std::string lora_adapter([_params.loraAdapter cStringUsingEncoding:NSUTF8StringEncoding]);
    std::string lora_base(_params.loraBase.length > 0 ? [_params.loraBase cStringUsingEncoding:NSUTF8StringEncoding] : "");
    int err = llama_apply_lora_from_file(ctx,
                                          lora_adapter.c_str(),
                                          lora_base.empty() ? NULL : lora_base.c_str(),
                                          _params.numberOfThreads);
    if (err != 0) {
      if (outError) {
        *outError = makeFailedToPredictErrorWithUnderlyingError(makeLlamaError(_LlamaErrorCodeFailedToApplyLoraAdapter, [NSString stringWithFormat:@"failed to apply lora adapter"]));
      }
      return NO;
    }
  }

  LlamaContext *context = [[LlamaContext alloc] initWithParams:_params context:ctx];

  auto runState = context.runState;
  runState->path_session = ""; //params.path_prompt_cache;
//
//  if (!path_session.empty()) {
//      fprintf(stderr, "%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
//
//      // fopen to check for existing session
//      FILE * fp = std::fopen(path_session.c_str(), "rb");
//      if (fp != NULL) {
//          std::fclose(fp);
//
//          session_tokens.resize(params.n_ctx);
//          size_t n_token_count_out = 0;
//          if (!llama_load_session_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
//              fprintf(stderr, "%s: error: failed to load session file '%s'\n", __func__, path_session.c_str());
//              return 1;
//          }
//          session_tokens.resize(n_token_count_out);
//          llama_set_rng_seed(ctx, params.seed);
//
//          fprintf(stderr, "%s: loaded a session with prompt size of %d tokens\n", __func__, (int) session_tokens.size());
//      } else {
//          fprintf(stderr, "%s: session file does not exist, will create\n", __func__);
//      }
//  }

  NSString *initialPrompt = @"";
  if (context.params.initialPrompt) {
    initialPrompt = context.params.initialPrompt;
  }

  std::string prompt([initialPrompt cStringUsingEncoding:NSUTF8StringEncoding]);

  // Add a space in front of the first character to match OG llama tokenizer behavior
  prompt.insert(0, 1, ' ');

  if (![LlamaOperationUtils tokenizeString:prompt with:context into:runState->embd_inp addBeginningOfSequence:true outError:outError]) {
    return NO;
  }

  const int n_ctx = llama_n_ctx(context.ctx);

  // debug message about similarity of saved session, if applicable
  size_t n_matching_session_tokens = 0;
  if (runState->session_tokens.size()) {
    for (llama_token id : runState->session_tokens) {
      if (n_matching_session_tokens >= runState->embd_inp.size() || id != runState->embd_inp[n_matching_session_tokens]) {
        break;
      }
      n_matching_session_tokens++;
    }
    if (prompt.empty() && n_matching_session_tokens == context.runState->embd_inp.size()) {
      fprintf(stderr, "%s: using full prompt from session file\n", __func__);
    } else if (n_matching_session_tokens >= context.runState->embd_inp.size()) {
      fprintf(stderr, "%s: session file has exact match for prompt!\n", __func__);
    } else if (n_matching_session_tokens < (context.runState->embd_inp.size() / 2)) {
      fprintf(stderr, "%s: warning: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
              __func__, n_matching_session_tokens, context.runState->embd_inp.size());
    } else {
      fprintf(stderr, "%s: session file matches %zu / %zu tokens of prompt\n",
              __func__, n_matching_session_tokens, context.runState->embd_inp.size());
    }
  }

  // if we will use the cache for the full prompt without reaching the end of the cache, force
  // reevaluation of the last token token to recalculate the cached logits
  if (!runState->embd_inp.empty() && n_matching_session_tokens == runState->embd_inp.size() &&
          runState->session_tokens.size() > runState->embd_inp.size()) {
      runState->session_tokens.resize(runState->embd_inp.size() - 1);
  }

  // number of tokens to keep when resetting context
  if (_params.numberOfTokensToKeepFromInitialPrompt < 0 || _params.numberOfTokensToKeepFromInitialPrompt > (int)context.runState->embd_inp.size()) {
    _params.numberOfTokensToKeepFromInitialPrompt = (int)context.runState->embd_inp.size();
  }

  runState->is_antiprompt = false;
  runState->need_to_save_session = !runState->path_session.empty() && n_matching_session_tokens < runState->embd_inp.size();

  runState->n_past = 0;
  runState->n_remain = context.params.numberOfTokens;
  runState->n_consumed = 0;
  runState->n_session_consumed = 0;

  if (outContext != NULL) {
    *outContext = context;
  }

  return YES;
}

@end
