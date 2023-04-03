//
//  LlamaSessionContext.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 02/04/2023.
//

#import "LlamaSessionContext.h"

static NSArray *deepCopyTokens(NSArray<NSNumber *> *tokens) {
  NSMutableArray *deepCopiedTokens = [[NSMutableArray alloc] initWithCapacity:tokens.count];
  for (NSNumber *token in tokens) {
    [deepCopiedTokens addObject:[token copy]];
  }
  return [deepCopiedTokens copy];
}

@implementation _LlamaSessionContext

- (instancetype)initWithContextString:(NSString *__nullable)contextString tokens:(NSArray<NSNumber *> *__nullable)tokens
{
  if ((self = [super init])) {
    _contextString = [contextString copy];
    _tokens = deepCopyTokens(tokens);
  }
  return self;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
  return [[_LlamaSessionContext alloc] initWithContextString:_contextString tokens:_tokens];
}

@end
