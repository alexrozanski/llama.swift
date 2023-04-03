//
//  LlamaSessionContext.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 02/04/2023.
//

#import "LlamaSessionContext.h"

static NSArray *deepCopyTokens(NSArray<_LlamaSessionContextToken *> *tokens) {
  NSMutableArray *deepCopiedTokens = [[NSMutableArray alloc] initWithCapacity:tokens.count];
  for (NSNumber *token in tokens) {
    [deepCopiedTokens addObject:[token copy]];
  }
  return [deepCopiedTokens copy];
}

@implementation _LlamaSessionContextToken

- (nonnull instancetype)initWithToken:(int)token string:(NSString *__nonnull)string
{
  if ((self = [super init])) {
    _token = token;
    _string = [string copy];
  }
  return self;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
  return [[_LlamaSessionContextToken alloc] initWithToken:_token string:_string];
}

@end

@implementation _LlamaSessionContext {
  NSString *_contextString;
}

- (instancetype)initWithTokens:(NSArray<_LlamaSessionContextToken *> *)tokens
{
  if ((self = [super init])) {
    _tokens = deepCopyTokens(tokens);
  }
  return self;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
  return [[_LlamaSessionContext alloc] initWithTokens:_tokens];
}

- (NSString *)contextString
{
  if (_contextString != nil) {
    return _contextString;
  }

  NSMutableString *contextString = [[NSMutableString alloc] init];
  for (_LlamaSessionContextToken *token in _tokens) {
    if (token.string != nil) {
      [contextString appendString:token.string];
    }
  }

  _contextString = [contextString copy];

  return _contextString;
}

@end
