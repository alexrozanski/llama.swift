//
//  LlamaErrorInternal.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 23/03/2023.
//

#import "LlamaErrorInternal.h"

NSError *makeLlamaError(LlamaErrorCode errorCode, NSString *description)
{
  return [[NSError alloc] initWithDomain:LlamaErrorDomain code:errorCode userInfo:@{
    NSLocalizedDescriptionKey: description
  }];
}
