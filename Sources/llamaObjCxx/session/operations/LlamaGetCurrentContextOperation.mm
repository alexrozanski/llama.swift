//
//  LlamaGetCurrentContextOperation.mm
//  llamaObjCxx
//
//  Created by Alex Rozanski on 02/04/2023.
//

#import "LlamaGetCurrentContextOperation.hh"

#import "LlamaContext.hh"
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

  NSMutableString *contextString = [[NSMutableString alloc] init];
  NSMutableArray<NSNumber *> *tokens = [[NSMutableArray alloc] init];

  for (auto &token : _context.runState->last_n_tokens) {
    if (token == 0) { continue; }

    const char *string = llama_token_to_str(_context.ctx, token);
    [contextString appendString:[NSString stringWithCString:string encoding:NSUTF8StringEncoding]];
    [tokens addObject:@(token)];
  }

  return [[_LlamaSessionContext alloc] initWithContextString:contextString tokens:tokens];
}

@end
