//
//  LlamaGetCurrentContextOperation.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 02/04/2023.
//

#import <Foundation/Foundation.h>

@class LlamaContext;

NS_ASSUME_NONNULL_BEGIN

typedef void (^LlamaGetContextOperationContextHandler)(NSString *context);

@interface LlamaGetCurrentContextOperation : NSOperation

- (instancetype)initWithContext:(LlamaContext *)context
           returnContextHandler:(LlamaGetContextOperationContextHandler)contextHandler
                   handlerQueue:(dispatch_queue_t)eventHandlerQueue;

@end

NS_ASSUME_NONNULL_END
