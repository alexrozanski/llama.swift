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
  }

  dispatch_async(dispatch_get_main_queue(), ^{
    [self->_delegate setupOperation:self didSucceedWithContext:context];
  });
}

- (BOOL)_setUpReturningContext:(LlamaContext **)outContext error:(NSError **)outError
{
  if (_params.contextSize > 2048) {
    NSLog(@"warning: model does not support context sizes greater than 2048 tokens (%d specified);"
          "expect poor results", _params.contextSize);
  }

  llama_context * ctx;

  // load the model
  {
    auto lparams = llama_context_default_params();

    lparams.n_ctx      = _params.contextSize;
    lparams.n_parts    = _params.numberOfParts;
    lparams.seed       = _params.seed;
    lparams.f16_kv     = _params.useF16Memory;
    lparams.use_mlock  = _params.keepModelInMemory;

    const char *modelPath = [_params.modelPath cStringUsingEncoding:NSUTF8StringEncoding];
    ctx = llama_init_from_file(modelPath, lparams, outError);
    if (ctx == NULL) {
      return NO;
    }
  }

  LlamaContext *context = [[LlamaContext alloc] initWithParams:_params context:ctx];
  NSString *initialPrompt = @"";
  if (context.params.initialPrompt) {
    initialPrompt = context.params.initialPrompt;
  }

  std::string prompt([initialPrompt cStringUsingEncoding:NSUTF8StringEncoding]);

  // Add a space in front of the first character to match OG llama tokenizer behavior
  prompt.insert(0, 1, ' ');

  // tokenize the initial prompt
  if (![LlamaOperationUtils tokenizeString:prompt with:context into:context.runState->embd_inp addBeginningOfSequence:true outError:outError]) {
    return NO;
  }

  // Initialize the run state.
  const int n_ctx = llama_n_ctx(context.ctx);

  // Remaining setup.
  if ((int)context.runState->embd_inp.size() > n_ctx - 4) {
    if (outError) {
      *outError = makeLlamaError(_LlamaErrorCodePredictionFailed, [NSString stringWithFormat:@"prompt is too long (%d tokens, max %d)\n", (int)context.runState->embd_inp.size(), n_ctx - 4]);
    }
    return NO;
  }

  auto runState = context.runState;

  // number of tokens to keep when resetting context
  if (context.params.numberOfTokensToKeepFromInitialPrompt < 0 || context.params.numberOfTokensToKeepFromInitialPrompt > (int)context.runState->embd_inp.size() || _params.isInstructional) {
    context.params.numberOfTokensToKeepFromInitialPrompt = (int)context.runState->embd_inp.size();
  }

  // TODO: replace with ring-buffer
  context.runState->last_n_tokens.resize(n_ctx);
  std::fill(context.runState->last_n_tokens.begin(), context.runState->last_n_tokens.end(), 0);

  runState->n_past = 0;
  runState->n_remain = context.params.numberOfTokens;
  runState->n_consumed = 0;

  runState->is_antiprompt = false;

  if (outContext != NULL) {
    *outContext = context;
  }

  return YES;
}

@end
