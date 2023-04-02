//
//  LlamaSession.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import <Foundation/Foundation.h>
#import "LlamaSessionPredictionHandle.h"

@class _LlamaSessionParams;

NS_ASSUME_NONNULL_BEGIN

@class _LlamaSession;

@protocol _LlamaSessionDelegate <NSObject>

- (void)didStartLoadingModelInSession:(_LlamaSession *)session;
- (void)didLoadModelInSession:(_LlamaSession *)session;
- (void)didStartPredictingInSession:(_LlamaSession *)session;
- (void)didFinishPredictingInSession:(_LlamaSession *)session;

- (void)session:(_LlamaSession *)session didMoveToErrorStateWithError:(NSError *)error;

@end

@interface _LlamaSession : NSObject

@property (nullable, weak, readonly) id<_LlamaSessionDelegate> delegate;

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

- (instancetype)initWithParams:(_LlamaSessionParams *)params delegate:(id<_LlamaSessionDelegate>)delegate;

// MARK: - Preloading

- (void)loadModelIfNeeded;

// MARK: - Prediction

- (id<_LlamaSessionPredictionHandle>)runPredictionWithPrompt:(NSString*)prompt
                                                startHandler:(void(^)(void))startHandler
                                                tokenHandler:(void(^)(NSString*))tokenHandler
                                           completionHandler:(void(^)(void))completionHandler
                                               cancelHandler:(void(^)(void))cancelHandler
                                              failureHandler:(void(^)(NSError*))errorHandler
                                                handlerQueue:(dispatch_queue_t)handlerQueue;

// MARK: - Diagnostics

- (void)getCurrentContextWithHandler:(void(^)(NSString *context))handler handlerQueue:(dispatch_queue_t)handlerQueue;

@end

NS_ASSUME_NONNULL_END
