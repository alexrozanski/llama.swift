//
//  LlamaSessionConcretePredictionHandle.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 31/03/2023.
//

#import "LlamaSessionConcretePredictionHandle.h"

@implementation _LlamaSessionConcretePredictionHandle

- (instancetype)initWithCancelHandler:(void (^)(void))cancelHandler
{
  if ((self = [super init])) {
    _cancelHandler = cancelHandler;
  }
  return self;
}

- (void)cancel
{
  if (_cancelHandler != NULL) {
    _cancelHandler();
  }
}

@end
