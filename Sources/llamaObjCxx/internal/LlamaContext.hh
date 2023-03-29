//
//  LlamaContext.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import <Foundation/Foundation.h>
#include <vector>

#import "common.hh"
#import "llama.hh"
#import "LlamaRunState.h"

@class LlamaContext;

NS_ASSUME_NONNULL_BEGIN

@interface LlamaContext : NSObject

// Holds the gpt_params which are set from the first prompt and for the lifetime of the session.
@property (nonatomic, readonly, assign, nullable) gpt_params *params;

// Context from Llama internal implementation.
@property (nonatomic, readonly, assign, nullable) llama_context *ctx;

// Run state shared between run invocations.
@property (nonatomic, readonly, assign, nullable) llama_swift_run_state *runState;

@property (nonatomic, readonly, getter=hasInitialized) BOOL initialized;

- (instancetype)initWithParams:(gpt_params)params context:(llama_context *)ctx;

- (BOOL)initializeWithInitializationBlock:(NS_NOESCAPE BOOL (^)(LlamaContext *, NSError **))initializationBlock
                                outError:(NSError **)outError;

@end

NS_ASSUME_NONNULL_END
