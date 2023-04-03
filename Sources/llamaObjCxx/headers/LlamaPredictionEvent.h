//
//  LlamaPredictionEvent.h
//  llama
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import <Foundation/Foundation.h>

@class _LlamaSessionContext;

NS_ASSUME_NONNULL_BEGIN

@interface _LlamaPredictionEvent : NSObject

+ (instancetype)started;
+ (instancetype)outputTokenWithToken:(nonnull NSString *)token;
+ (instancetype)updatedSessionContext:(nonnull _LlamaSessionContext *)sessionContext;
+ (instancetype)completed;
+ (instancetype)cancelled;
+ (instancetype)failedWithError:(nonnull NSError *)error;

- (void)matchStarted:(void (^)(void))started
         outputToken:(void (^)(NSString *token))outputToken
updatedSessionContext:(void (^)(_LlamaSessionContext *sessionContext))updatedSessionContext
           completed:(void (^)(void))completed
           cancelled:(void (^)(void))cancelled
              failed:(void (^)(NSError *__nullable error))failed;

@end

NS_ASSUME_NONNULL_END
