//
//  LlamaGetCurrentContextOperation.mm
//  llamaObjCxx
//
//  Created by Alex Rozanski on 02/04/2023.
//

#import "LlamaGetCurrentContextOperation.hh"

#import "LlamaContext.hh"

@implementation LlamaGetCurrentContextOperation {
  LlamaContext *_context;
  LlamaGetContextOperationContextHandler _contextHandler;
  dispatch_queue_t _handlerQueue;
}

- (instancetype)initWithContext:(LlamaContext *)context
           returnContextHandler:(LlamaGetContextOperationContextHandler)contextHandler
                   handlerQueue:(dispatch_queue_t)handlerQueue
{
  if ((self = [super init])) {
    _context = context;
  }
  return self;
}

- (void)main
{
  NSMutableString *context = [[NSMutableString alloc] init];
  for (auto &token : _context.runState->last_n_tokens) {
    if (token == 0) { continue; }

    const char *string = llama_token_to_str(_context.ctx, token);
    [context appendString:[NSString stringWithCString:string encoding:NSUTF8StringEncoding]];
  }

  if (_contextHandler) {
    dispatch_async(_handlerQueue, ^{
      self->_contextHandler(context);
    });
  }
}

@end
