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

  // print system information
//  {
//    fprintf(stderr, "\n");
//    fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
//            params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
//  }

  LlamaContext *context = [[LlamaContext alloc] initWithContext:ctx];

  dispatch_async(dispatch_get_main_queue(), ^{
    [self->_delegate setupOperation:self didSucceedWithContext:context];
  });
}

@end
