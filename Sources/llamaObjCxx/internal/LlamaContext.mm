//
//  LlamaContext.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import "LlamaContext.hh"
#import "llama.hh"

@implementation LlamaContext

@synthesize ctx = _ctx;

- (instancetype)initWithContext:(llama_context *)ctx
{
  if ((self = [super init])) {
    _ctx = ctx;
  }
  return self;
}

- (void)dealloc
{
  llama_free(_ctx);
  _ctx = nullptr;
}

@end
