//
//  LlamaSessionConcretePredictionHandle.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 31/03/2023.
//

#import <Foundation/Foundation.h>
#import "LlamaSessionPredictionHandle.h"

NS_ASSUME_NONNULL_BEGIN

@interface _LlamaSessionConcretePredictionHandle : NSObject <_LlamaSessionPredictionHandle>

@property (nonatomic, readonly, copy) void (^cancelHandler)(void);

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithCancelHandler:(void (^)(void))cancelHandler;

@end

NS_ASSUME_NONNULL_END
