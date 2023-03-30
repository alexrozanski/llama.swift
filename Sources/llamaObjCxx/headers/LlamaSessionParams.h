//
//  LlamaSessionParams.h
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, _LlamaSessionMode) {
  _LlamaSessionModeRegular = 0,
  _LlamaSessionModeInstructional
};

@interface _LlamaSessionParams : NSObject

@property (nonatomic, readonly, assign) _LlamaSessionMode mode;

@property (nonatomic, nullable, copy) NSString *initialPrompt;
@property (nonatomic, nullable, copy) NSString *promptPrefix;
@property (nonatomic, nullable, copy) NSString *promptSuffix;

@property (nonatomic, assign) NSUInteger numberOfThreads;
@property (nonatomic, assign) NSUInteger numberOfTokens;

@property (nullable, copy) NSArray<NSString *> *antiprompts;
@property (nonatomic, assign) int32_t seed;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithMode:(_LlamaSessionMode)mode;

@end

NS_ASSUME_NONNULL_END
