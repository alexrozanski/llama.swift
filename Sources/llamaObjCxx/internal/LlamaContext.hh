//
//  LlamaContext.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import <Foundation/Foundation.h>
#include <vector>

#import "llama.hh"
#import "LlamaRunState.h"

@class _LlamaSessionParams;
@class LlamaContext;

NS_ASSUME_NONNULL_BEGIN

@interface LlamaContext : NSObject

@property (nonatomic, readonly) _LlamaSessionParams *params;

// Context from Llama internal implementation.
@property (nonatomic, readonly, assign, nullable) llama_context *ctx;

// Run state shared between run invocations.
@property (nonatomic, readonly, assign, nullable) llama_swift_run_state *runState;

@property (nonatomic, readonly, getter=hasInitialized) BOOL initialized;

- (instancetype)initWithParams:(_LlamaSessionParams *)params context:(llama_context *)ctx;

- (BOOL)initializeWithInitializationBlock:(NS_NOESCAPE BOOL (^)(LlamaContext *, NSError **))initializationBlock
                                outError:(NSError **)outError;

@end

NS_ASSUME_NONNULL_END
