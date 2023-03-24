//
//  LlamaRunnerBridge.h
//  llama
//
//  Created by Alex Rozanski on 12/03/2023.
//

#import <Foundation/Foundation.h>

@class _LlamaEvent;
@class _LlamaSessionConfig;

NS_ASSUME_NONNULL_BEGIN

typedef void (^_LlamaRunnerBridgeEventHandler)(_LlamaEvent *event);

@interface _LlamaRunnerBridge : NSObject

@property (nonnull, readonly, copy) NSString *modelPath;

- (instancetype)initWithModelPath:(nonnull NSString *)modelPath;

- (void)runWithPrompt:(nonnull NSString*)prompt
               config:(nonnull _LlamaSessionConfig *)config
         eventHandler:(nonnull _LlamaRunnerBridgeEventHandler)eventHandler
    eventHandlerQueue:(nonnull dispatch_queue_t)eventHandlerQueue;
@end

NS_ASSUME_NONNULL_END
