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

@class LlamaContext;

NS_ASSUME_NONNULL_BEGIN

@interface LlamaOperationUtils : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

// adapted from llama_tokenize() from common.h
+ (BOOL)tokenizeString:(const std::string &)string
                  with:(LlamaContext *)context
                  into:(std::vector<llama_token> &)tokens
addBeginningOfSequence:(bool)addBeginningOfSequence
              outError:(NSError **)outError;

@end

NS_ASSUME_NONNULL_END
