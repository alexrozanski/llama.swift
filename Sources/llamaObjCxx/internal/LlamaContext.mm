//
//  LlamaContext.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import "LlamaContext.hh"

#import "LlamaSessionParams.h"

@implementation LlamaContext

@synthesize params = _params;
@synthesize ctx = _ctx;
@synthesize runState = _runState;

- (instancetype)initWithParams:(_LlamaSessionParams *)params context:(llama_context *)ctx
{
  if ((self = [super init])) {
    _params = params;
    _ctx = ctx;
    _runState = new llama_swift_run_state;
  }
  return self;
}

- (void)dealloc
{
  llama_free(_ctx);
  _ctx = nullptr;

  delete _runState;
  _runState = nullptr;
}

@end
