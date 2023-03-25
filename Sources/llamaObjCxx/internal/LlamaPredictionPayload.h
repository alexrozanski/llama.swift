//
//  LlamaPredictionPayload.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 25/03/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface LlamaPredictionPayload : NSObject

@property (nonatomic, readonly, copy, nonnull) NSString *prompt;
@property (nonatomic, nullable) void (^tokenHandler)(NSString*);
@property (nonatomic, nullable) void (^completionHandler)(void);
@property (nonatomic, nullable) void (^failureHandler)(NSError*);

- (instancetype)initWithPrompt:(NSString *)prompt
                  tokenHandler:(void(^)(NSString*))tokenHandler
             completionHandler:(void(^)(void))completionHandler
                failureHandler:(void(^)(NSError*))failureHandler;

@end

NS_ASSUME_NONNULL_END
