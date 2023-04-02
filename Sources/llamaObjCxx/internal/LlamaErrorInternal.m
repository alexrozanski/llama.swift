//
//  LlamaErrorInternal.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 23/03/2023.
//

#import "LlamaErrorInternal.h"

NSError *makeLlamaError(_LlamaErrorCode errorCode, NSString *description)
{
  return [[NSError alloc] initWithDomain:_LlamaErrorDomain code:errorCode userInfo:@{
    NSLocalizedDescriptionKey: description
  }];
}
