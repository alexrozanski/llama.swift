//
//  LlamaError.h
//  llama
//
//  Created by Alex Rozanski on 14/03/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

extern NSString *const _LlamaErrorDomain;

typedef NS_ENUM(NSInteger, _LlamaErrorCode) {
  _LlamaErrorCodeUnknown = -1,

  _LlamaErrorCodeFailedToLoadModel = -1000,
  _LlamaErrorCodeFailedToPredict = -1001,
  _LlamaErrorCodeFailedToLoadSessionContext = -1002,
};

NS_ASSUME_NONNULL_END
