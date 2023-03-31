//
//  LlamaPredictionPayload.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 25/03/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface LlamaPredictionPayload : NSObject

@property (nonatomic, readonly, copy) NSString *identifier;
@property (nonatomic, readonly, copy) NSString *prompt;
@property (nonatomic, readonly, copy) void (^startHandler)(void);
@property (nonatomic, readonly, copy) void (^tokenHandler)(NSString*);
@property (nonatomic, readonly, copy) void (^completionHandler)(void);
@property (nonatomic, readonly, copy) void (^cancelHandler)(void);
@property (nonatomic, readonly, copy) void (^failureHandler)(NSError*);
@property (nonatomic, readonly, assign) dispatch_queue_t handlerQueue;

- (instancetype)initWithIdentifier:(NSString *)identifier
                            prompt:(NSString *)prompt
                      startHandler:(void(^)(void))startHandler
                      tokenHandler:(void(^)(NSString*))tokenHandler
                 completionHandler:(void(^)(void))completionHandler
                     cancelHandler:(void(^)(void))cancelHandler
                    failureHandler:(void(^)(NSError*))failureHandler
                      handlerQueue:(dispatch_queue_t)handlerQueue;

@end

NS_ASSUME_NONNULL_END
