//
//  LlamaGetCurrentContextOperation.mm
//  llamaObjCxx
//
//  Created by Alex Rozanski on 02/04/2023.
//

#import "LlamaGetCurrentContextOperation.hh"

#import "LlamaContext.hh"
#import "LlamaOperationUtils.hh"
#import "LlamaSessionContext.h"
#import "LlamaSessionContext+Internal.h"

@implementation LlamaGetCurrentContextOperation {
  LlamaContext *_context;
  LlamaGetContextOperationContextHandler _contextHandler;
}

- (instancetype)initWithContext:(LlamaContext *)context returnContextHandler:(LlamaGetContextOperationContextHandler)contextHandler
{
  if ((self = [super init])) {
    _context = context;
    _contextHandler = [contextHandler copy];
  }
  return self;
}

- (void)main
{
  _LlamaSessionContext *context = [self _run];
  if (_contextHandler) {
    self->_contextHandler(context);
  }
}

- (_LlamaSessionContext *)_run
{
  if (_context.ctx == NULL || _context.runState == NULL) {
    return nil;
  }

  return [LlamaOperationUtils currentSessionContextWithLlamaContext:_context];
}

@end
