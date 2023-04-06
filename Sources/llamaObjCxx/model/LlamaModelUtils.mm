//
//  LlamaModelUtils.mm
//  llamaObjCxx
//
//  Created by Alex Rozanski on 05/04/2023.
//

#import "LlamaModelUtils.h"

#import "llama.hh"

@implementation _LlamaModelUtils

+ (BOOL)loadModelTypeForFileAtURL:(NSURL *)fileURL
                     outModelType:(_LlamaModelType *)outModelType
                         outError:(NSError **)outError
{
  const char *path = [[fileURL path] cStringUsingEncoding:NSUTF8StringEncoding];
  e_model model_type;
  if (!llama_get_model_type(path, model_type, outError)) {
    return NO;
  }

  if (outModelType) {
    switch (model_type) {
      case ::MODEL_UNKNOWN:
        *outModelType = _LlamaModelTypeUnknown;
        break;
      case ::MODEL_7B:
        *outModelType = _LlamaModelType7B;
        break;
      case ::MODEL_13B:
        *outModelType = _LlamaModelType13B;
        break;
      case ::MODEL_30B:
        *outModelType = _LlamaModelType30B;
        break;
      case ::MODEL_65B:
        *outModelType = _LlamaModelType65B;
        break;
      default:
        break;
    }
  }

  return YES;
}

@end
