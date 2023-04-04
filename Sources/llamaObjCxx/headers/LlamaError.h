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

  // High-level error codes
  _LlamaErrorCodeFailedToValidateModel = -100,
  _LlamaErrorCodeFailedToLoadModel = -101,
  _LlamaErrorCodeFailedToPredict = -102,
  _LlamaErrorCodeFailedToLoadSessionContext = -103,

  // Model internal error codes
  _LlamaErrorCodeFailedToOpenModelFile = -1000,
  _LlamaErrorCodeInvalidModelUnversioned = -1001,
  _LlamaErrorCodeInvalidModelBadMagic = -1002,
  _LlamaErrorCodeInvalidModelUnsupportedFileVersion = -1003,

  // Model prediction internal error codes
  _LlamaErrorCodePromptIsTooLong = -2000,

  // General failure error codes
  _LlamaErrorCodeGeneralInternalLoadFailure = -10001,
  _LlamaErrorCodeGeneralInternalPredictionFailure = -10002,
};

NS_ASSUME_NONNULL_END
