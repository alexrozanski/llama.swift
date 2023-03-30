//
//  LlamaSetupOperation.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 23/03/2023.
//

#import "LlamaSetupOperation.hh"

#import "LlamaContext.hh"
#import "LlamaError.h"
#import "LlamaEvent.h"
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
  if (_params.contextSize > 2048) {
    NSLog(@"warning: model does not support context sizes greater than 2048 tokens (%d specified);"
          "expect poor results\n", _params.contextSize);
  }

  llama_context * ctx;

  // load the model
  {
    auto lparams = llama_context_default_params();

    lparams.n_ctx      = _params.contextSize;
    lparams.n_parts    = _params.numberOfParts;
    lparams.seed       = _params.seed;
    lparams.f16_kv     = _params.useF16Memory;

    // Expose perplexity on params?
    lparams.logits_all = false; // _params.perplexity;

    NSError *loadError = nil;
    const char *modelPath = [_params.modelPath cStringUsingEncoding:NSUTF8StringEncoding];
    ctx = llama_init_from_file(modelPath, lparams, &loadError);

    if (ctx == NULL) {
      dispatch_async(dispatch_get_main_queue(), ^{
        [self->_delegate setupOperation:self didFailWithError:loadError];
      });
      return;
    }
  }

  dispatch_async(dispatch_get_main_queue(), ^{
    LlamaContext *context = [[LlamaContext alloc] initWithParams:self->_params context:ctx];
    [self->_delegate setupOperation:self didSucceedWithContext:context];
  });
}

@end
