//
//  LlamaOperationUtils.hh
//  llamaObjCxx
//
//  Created by Alex Rozanski on 01/04/2023.
//

#import <Foundation/Foundation.h>

#include <string>
#include <vector>

#include "llama.hh"

@class _LlamaSessionContext;
@class LlamaContext;

NS_ASSUME_NONNULL_BEGIN

/* WARNING: These methods should only be run on the session operation queue to ensure correct access of LlamaContext. */
@interface LlamaOperationUtils : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

// adapted from llama_tokenize() from common.h
+ (BOOL)tokenizeString:(const std::string &)string
                  with:(LlamaContext *)context
                  into:(std::vector<llama_token> &)tokens
addBeginningOfSequence:(bool)addBeginningOfSequence
              outError:(NSError **)outError;

+ (_LlamaSessionContext *)currentSessionContextWithLlamaContext:(LlamaContext *)context;

@end

NS_ASSUME_NONNULL_END
