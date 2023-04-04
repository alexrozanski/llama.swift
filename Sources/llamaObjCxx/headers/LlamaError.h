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
  _LlamaErrorCodeFailedToLoadModel = -1000,
  _LlamaErrorCodeFailedToPredict = -1001,
  _LlamaErrorCodeFailedToLoadSessionContext = -1002,

  // Model failure internal error codes
  _LlamaErrorCodeFailedToOpenModelFile = -2000,
  _LlamaErrorCodeInvalidModelUnversioned = -2001,
  _LlamaErrorCodeInvalidModelBadMagic = -2002,
  _LlamaErrorCodeInvalidModelUnsupportedFileVersion = -2003,

  // Model prediction internal error codes
  _LlamaErrorCodePromptIsTooLong = -3000,

  // General failure error codes
  _LlamaErrorCodeGeneralInternalLoadFailure = -10000,
  _LlamaErrorCodeGeneralInternalPredictionFailure = -10001,
};

NS_ASSUME_NONNULL_END
