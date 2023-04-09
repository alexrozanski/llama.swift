//
//  LlamaModelUtils.mm
//  llamaObjCxx
//
//  Created by Alex Rozanski on 05/04/2023.
//

#import "LlamaModelUtils.h"

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
                      quantizationType:(_LlamaQuantizationType)quantizationType
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

  int itype = 2;
  switch (quantizationType) {
  case _LlamaQuantizationTypeQ4_1:
    itype = 3;
  case _LlamaQuantizationTypeUnknown:
  case _LlamaQuantizationTypeQ4_0:
    break;
  }

  if (!llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), itype, outError)) {
    return NO;
  }

  return YES;
}

@end
