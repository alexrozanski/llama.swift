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

@interface LlamaSetupOperation () {
  gpt_params _params;
  LlamaSetupOperationEventHandler _eventHandler;
  dispatch_queue_t _eventHandlerQueue;
}

@end

@implementation LlamaSetupOperation

@synthesize delegate = _delegate;

- (instancetype)initWithParams:(gpt_params)params delegate:(id<LlamaSetupOperationDelegate>)delegate
{
  if ((self = [super init])) {
    _params = params;
    _delegate = delegate;
  }

  return self;
}

- (void)main
{
  gpt_params params(_params);

  if (params.n_ctx > 2048) {
    NSLog(@"warning: model does not support context sizes greater than 2048 tokens (%d specified);"
          "expect poor results\n", params.n_ctx);
  }

  llama_context * ctx;

  // load the model
  {
    auto lparams = llama_context_default_params();

    lparams.n_ctx      = _params.n_ctx;
    lparams.n_parts    = _params.n_parts;
    lparams.seed       = _params.seed;
    lparams.f16_kv     = _params.memory_f16;
    lparams.logits_all = _params.perplexity;

    NSError *loadError = nil;
    ctx = llama_init_from_file(_params.model.c_str(), lparams, &loadError);

    if (ctx == NULL) {
      dispatch_async(dispatch_get_main_queue(), ^{
        [self->_delegate setupOperation:self didFailWithError:loadError];
      });
      return;
    }
  }

  dispatch_async(dispatch_get_main_queue(), ^{
    LlamaContext *context = [[LlamaContext alloc] initWithParams:params context:ctx];
    [self->_delegate setupOperation:self didSucceedWithContext:context];
  });
}

@end
