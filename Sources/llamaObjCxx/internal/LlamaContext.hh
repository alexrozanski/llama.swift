//
//  LlamaContext.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import <Foundation/Foundation.h>

#import "llama.hh"

NS_ASSUME_NONNULL_BEGIN

@interface LlamaContext : NSObject

@property (readonly, assign, nullable) llama_context *ctx;

- (instancetype)initWithContext:(llama_context *)ctx;

@end

NS_ASSUME_NONNULL_END
