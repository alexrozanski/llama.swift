//
//  LlamaModelUtils.mm
//  llamaObjCxx
//
//  Created by Alex Rozanski on 05/04/2023.
//

#import "LlamaModelUtils.h"

#include <string>

#import "llama.hh"
#import "LlamaErrorInternal.h"

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

+ (BOOL)quantizeModelWithSourceFileURL:(NSURL *)fileURL
                           destFileURL:(NSURL *)destFileURL
                              fileType:(_LlamaModelFileType)fileType
                              outError:(NSError **)outError
{
  if (!fileURL) {
    if (outError) {
      *outError = makeFailedToQuantizeErrorWithUnderlyingError(makeLlamaError(_LlamaErrorCodeInvalidInputArguments, @"Missing source file path"));
    }
    return NO;
  }

  if (!destFileURL) {
    if (outError) {
      *outError = makeFailedToQuantizeErrorWithUnderlyingError(makeLlamaError(_LlamaErrorCodeInvalidInputArguments, @"Missing destination file path"));
    }
    return NO;
  }

  const std::string fname_inp([fileURL.path cStringUsingEncoding:NSUTF8StringEncoding]);
  const std::string fname_out([destFileURL.path cStringUsingEncoding:NSUTF8StringEncoding]);

  llama_ftype ftype;
  switch (fileType) {
  case _LlamaModelFileTypeUnknown:
      if (outError) {
        *outError = makeFailedToQuantizeErrorWithUnderlyingError(makeLlamaError(_LlamaErrorCodeInvalidInputArguments, @"Invalid input fileType"));
      }
      return NO;
  case _LlamaModelFileTypeAllF32:
      ftype = LLAMA_FTYPE_ALL_F32;
      break;
  case _LlamaModelFileTypeMostlyF16:
      ftype = LLAMA_FTYPE_MOSTLY_F16;
      break;
  case _LlamaModelFileTypeMostlyQ4_0:
      ftype = LLAMA_FTYPE_MOSTLY_Q4_0;
      break;
  case _LlamaModelFileTypeMostlyQ4_1:
      ftype = LLAMA_FTYPE_MOSTLY_Q4_1;
      break;
  case _LlamaModelFileTypeMostlyQ4_1SomeF16:
      ftype = LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16;
      break;
  }

  if (!llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), ftype, outError)) {
    return NO;
  }

  return YES;
}

@end
