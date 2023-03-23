//
//  LlamaPredictOperation.h
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import <Foundation/NSOperation.h>
#import "utils.hh"

@class _LlamaEvent;

NS_ASSUME_NONNULL_BEGIN

typedef void (^LlamaPredictOperationEventHandler)(_LlamaEvent *event);

@interface LlamaPredictOperation : NSOperation

- (instancetype)initWithParams:(gpt_params)params
                  eventHandler:(LlamaPredictOperationEventHandler)eventHandler
             eventHandlerQueue:(dispatch_queue_t)eventHandlerQueue;

@end

NS_ASSUME_NONNULL_END
