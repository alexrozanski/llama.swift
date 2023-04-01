//
//  LlamaOperationUtils.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 01/04/2023.
//

#import "LlamaOperationUtils.hh"

#import "LlamaContext.hh"

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

@end
