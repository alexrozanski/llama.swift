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
  std::vector<llama_token> res(string.size() + (int)addBeginningOfSequence);
  int n = llama_tokenize(context.ctx, string.c_str(), res.data(), (int)res.size(), addBeginningOfSequence, outError);
  if (n < 0) {
    return NO;
  }
  res.resize(n);
  tokens = res;
  return YES;
}

+ (_LlamaSessionContext *)currentSessionContextWithLlamaContext:(LlamaContext *)context
{
  if (context.ctx == NULL || context.runState == NULL) {
    return nil;
  }

  NSMutableString *contextString = [[NSMutableString alloc] init];
  NSMutableArray<NSNumber *> *tokens = [[NSMutableArray alloc] init];

  for (auto &token : context.runState->last_n_tokens) {
    if (token == 0) { continue; }

    const char *string = llama_token_to_str(context.ctx, token);
    [contextString appendString:[NSString stringWithCString:string encoding:NSUTF8StringEncoding]];
    [tokens addObject:@(token)];
  }

  return [[_LlamaSessionContext alloc] initWithContextString:contextString tokens:tokens];
}

@end
