//
//  LlamaErrorInternal.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 23/03/2023.
//

#import "LlamaErrorInternal.h"

NSError *__nonnull makeLlamaError(_LlamaErrorCode errorCode, NSString *__nonnull description)
{
  return makeLlamaErrorWithUnderlyingError(errorCode, description, nil);
}

NSError *makeLlamaErrorWithUnderlyingError(_LlamaErrorCode errorCode, NSString *description, NSError *underlyingError)
{
  NSMutableDictionary *userInfo = [[NSMutableDictionary alloc] init];
  if (description != nil) {
    userInfo[NSLocalizedDescriptionKey] = description;
  }

  if (underlyingError != nil) {
    userInfo[NSUnderlyingErrorKey] = underlyingError;
  }

  return [[NSError alloc] initWithDomain:_LlamaErrorDomain code:errorCode userInfo:userInfo];
}

NSError *makeFailedToLoadModelErrorWithUnderlyingError(NSError *underlyingError)
{
  return makeLlamaErrorWithUnderlyingError(_LlamaErrorCodeFailedToLoadModel, @"Failed to load model", underlyingError);
}

NSError *makeFailedToPredictErrorWithUnderlyingError(NSError *underlyingError)
{
  return makeLlamaErrorWithUnderlyingError(_LlamaErrorCodeFailedToPredict, @"Failed to run prediction", underlyingError);
}
