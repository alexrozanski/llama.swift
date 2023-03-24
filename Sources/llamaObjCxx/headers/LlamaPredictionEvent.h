//
//  LlamaPredictionEvent.h
//  llama
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface _LlamaPredictionEvent : NSObject

+ (instancetype)outputTokenWithToken:(nonnull NSString *)token;
+ (instancetype)completed;
+ (instancetype)failedWithError:(nonnull NSError *)error;

- (void)matchOutputToken:(void (^)(NSString *token))outputToken
               completed:(void (^)(void))completed
                  failed:(void (^)(NSError *error))failed;

@end

NS_ASSUME_NONNULL_END
