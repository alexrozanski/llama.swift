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
  _LlamaErrorCodeFailedToQuantize = -104,

  // General error codes
  _LlamaErrorCodeInvalidInputArguments = -500,

  // Model internal error codes
  _LlamaErrorCodeFailedToGetModelType = -1000,
  _LlamaErrorCodeFailedToOpenModelFile = -1001,
  _LlamaErrorCodeInvalidModelUnversioned = -1002,
  _LlamaErrorCodeInvalidModelBadMagic = -1003,
  _LlamaErrorCodeInvalidModelUnsupportedFileVersion = -1004,

  // Model prediction internal error codes
  _LlamaErrorCodePromptIsTooLong = -2000,
  _LlamaErrorCodeFailedToApplyLoraAdapter = -2001,

  // General failure error codes
  _LlamaErrorCodeGeneralInternalLoadFailure = -10001,
  _LlamaErrorCodeGeneralInternalPredictionFailure = -10002,
  _LlamaErrorCodeGeneralInternalQuantizationFailure = -10004,
};

NS_ASSUME_NONNULL_END
