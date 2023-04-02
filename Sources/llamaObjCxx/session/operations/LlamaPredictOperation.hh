//
//  LlamaPredictOperation.h
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import <Foundation/NSOperation.h>

@class LlamaContext;
@class _LlamaPredictionEvent;

NS_ASSUME_NONNULL_BEGIN

typedef void (^LlamaPredictOperationEventHandler)(_LlamaPredictionEvent *event);

@interface LlamaPredictOperation : NSOperation

@property (nonatomic, readonly, copy) NSString *identifier;

- (instancetype)initWithIdentifier:(NSString *)identifier
                           context:(LlamaContext *)context
                            prompt:(NSString *)prompt
                      eventHandler:(LlamaPredictOperationEventHandler)eventHandler;

@end

NS_ASSUME_NONNULL_END
