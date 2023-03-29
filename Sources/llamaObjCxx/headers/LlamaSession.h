//
//  LlamaSession.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import <Foundation/Foundation.h>

@class _LlamaSessionConfig;

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

@property (nonnull, readonly, copy) NSString *modelPath;
@property (nullable, weak, readonly) id<_LlamaSessionDelegate> delegate;

- (instancetype)init NS_UNAVAILABLE;

- (instancetype)initWithModelPath:(NSString *)modelPath
                           config:(_LlamaSessionConfig *)config
                         delegate:(id<_LlamaSessionDelegate>)delegate;

- (void)loadModelIfNeeded;

- (void)runPredictionWithPrompt:(NSString*)prompt
                   tokenHandler:(void(^)(NSString*))tokenHandler
              completionHandler:(void(^)(void))completionHandler
                 failureHandler:(void(^)(NSError*))errorHandler;

@end

NS_ASSUME_NONNULL_END
