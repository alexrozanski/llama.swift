//
//  LlamaErrorInternal.h
//  llama
//
//  Created by Alex Rozanski on 23/03/2023.
//

#import <Foundation/Foundation.h>

#import "LlamaError.h"

#ifdef  __cplusplus
extern "C" {
#endif

NSError *__nonnull makeLlamaError(_LlamaErrorCode errorCode, NSString *__nonnull description);
NSError *__nonnull makeLlamaErrorWithUnderlyingError(_LlamaErrorCode errorCode, NSString *__nonnull description, NSError *__nullable underlyingError);

NSError *__nonnull makeFailedToLoadModelErrorWithUnderlyingError(NSError *__nullable underlyingError);
NSError *__nonnull makeFailedToPredictErrorWithUnderlyingError(NSError *__nullable underlyingError);
NSError *__nonnull makeFailedToQuantizeErrorWithUnderlyingError(NSError *__nullable underlyingError);

#ifdef  __cplusplus
}
#endif
