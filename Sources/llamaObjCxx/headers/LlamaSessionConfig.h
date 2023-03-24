//
//  LlamaSessionConfig.h
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface _LlamaSessionConfig : NSObject

@property (nonatomic, assign) NSUInteger numberOfThreads;
@property (nonatomic, assign) NSUInteger numberOfTokens;

@property (nullable, copy) NSString *reversePrompt;
@property (nonatomic, assign) int32_t seed;

@end

NS_ASSUME_NONNULL_END
