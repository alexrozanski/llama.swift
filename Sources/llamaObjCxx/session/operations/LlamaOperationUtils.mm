//
//  LlamaOperationUtils.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 01/04/2023.
//

#import "LlamaOperationUtils.hh"

#import "LlamaContext.hh"
#import "LlamaSessionContext.h"
#import "LlamaSessionContext+Internal.h"

@implementation LlamaOperationUtils

+ (BOOL)tokenizeString:(const std::string &)string
                  with:(LlamaContext *)context
                  into:(std::vector<llama_token> &)tokens
addBeginningOfSequence:(bool)addBeginningOfSequence
              outError:(NSError **)outError
{
  // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
  std::vector<llama_token> res(string.size() + (int) addBeginningOfSequence);
  const int n = llama_tokenize(context.ctx, string.c_str(), res.data(), res.size(), addBeginningOfSequence);
  assert(n >= 0);
  res.resize(n);

  tokens = res;

  return YES;
}

+ (_LlamaSessionContext *)currentSessionContextWithLlamaContext:(LlamaContext *)context
{
  if (context.ctx == NULL || context.runState == NULL) {
    return nil;
  }

  NSMutableArray<_LlamaSessionContextToken *> *tokens = [[NSMutableArray alloc] init];

  for (auto &token : context.runState->last_n_tokens) {
    if (token == 0) { continue; }

    const char *cString = llama_token_to_str(context.ctx, token);
    NSString *string = [NSString stringWithCString:cString encoding:NSUTF8StringEncoding];
    if (!string) {
      string = @"";
    }
    [tokens addObject:[[_LlamaSessionContextToken alloc] initWithToken:token string:string]];
  }

  return [[_LlamaSessionContext alloc] initWithTokens:tokens];
}

@end
