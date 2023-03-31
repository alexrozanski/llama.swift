//
//  LlamaPredictionPayload.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 25/03/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface LlamaPredictionPayload : NSObject

@property (nonatomic, readonly, copy, nonnull) NSString *identifier;
@property (nonatomic, readonly, copy, nonnull) NSString *prompt;
@property (nonatomic, nullable) void (^tokenHandler)(NSString*);
@property (nonatomic, nullable) void (^completionHandler)(void);
@property (nonatomic, nullable) void (^failureHandler)(NSError*);
@property (nonatomic, assign) dispatch_queue_t handlerQueue;

- (instancetype)initWithIdentifier:(NSString *)identifier
                            prompt:(NSString *)prompt
                      tokenHandler:(void(^)(NSString*))tokenHandler
                 completionHandler:(void(^)(void))completionHandler
                    failureHandler:(void(^)(NSError*))failureHandler
                      handlerQueue:(dispatch_queue_t)handlerQueue;

@end

NS_ASSUME_NONNULL_END
