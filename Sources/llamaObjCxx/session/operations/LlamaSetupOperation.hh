//
//  LlamaSetupOperation.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 23/03/2023.
//

#import <Foundation/Foundation.h>

#import "utils.hh"

@class _LlamaEvent;
@class LlamaContext;
@class LlamaSetupOperation;

NS_ASSUME_NONNULL_BEGIN

@protocol LlamaSetupOperationDelegate <NSObject>
- (void)setupOperation:(LlamaSetupOperation *)operation didSucceedWithContext:(LlamaContext *)context;
- (void)setupOperation:(LlamaSetupOperation *)operation didFailWithError:(NSError *)error;
@end

typedef void (^LlamaSetupOperationEventHandler)(_LlamaEvent *event);

@interface LlamaSetupOperation : NSOperation

@property (nonatomic, weak) id<LlamaSetupOperationDelegate> delegate;

- (instancetype)initWithParams:(gpt_params)params delegate:(id<LlamaSetupOperationDelegate>)delegate;

@end

NS_ASSUME_NONNULL_END
