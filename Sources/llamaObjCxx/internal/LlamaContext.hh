//
//  LlamaContext.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import <Foundation/Foundation.h>
#include <vector>

#import "llama.hh"
#import "llama_swift_run_state.h"

NS_ASSUME_NONNULL_BEGIN

@interface LlamaContext : NSObject

// Context from Llama internal implementation.
@property (readonly, assign, nullable) llama_context *ctx;

// Run state shared between run invocations.
@property (readonly, assign, nullable) llama_swift_run_state *runState;

- (instancetype)initWithContext:(llama_context *)ctx;

@end

NS_ASSUME_NONNULL_END
