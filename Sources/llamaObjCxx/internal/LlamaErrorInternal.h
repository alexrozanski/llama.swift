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

NSError *makeLlamaError(LlamaErrorCode errorCode, NSString *description);

#ifdef  __cplusplus
}
#endif
