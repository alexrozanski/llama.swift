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

struct quant_option {
    std::string name;
    llama_ftype ftype;
    std::string desc;
};

static const std::vector<struct quant_option> QUANT_OPTIONS = {
    {
        "Q4_0",
        LLAMA_FTYPE_MOSTLY_Q4_0,
        " 3.50G, +0.2499 ppl @ 7B - small, very high quality loss - legacy, prefer using Q3_K_M",
    },
    {
        "Q4_1",
        LLAMA_FTYPE_MOSTLY_Q4_1,
        " 3.90G, +0.1846 ppl @ 7B - small, substantial quality loss - legacy, prefer using Q3_K_L",
    },
    {
        "Q5_0",
        LLAMA_FTYPE_MOSTLY_Q5_0,
        " 4.30G, +0.0796 ppl @ 7B - medium, balanced quality - legacy, prefer using Q4_K_M",
    },
    {
        "Q5_1",
        LLAMA_FTYPE_MOSTLY_Q5_1,
        " 4.70G, +0.0415 ppl @ 7B - medium, low quality loss - legacy, prefer using Q5_K_M",
    },
#ifdef GGML_USE_K_QUANTS
    {
        "Q2_K",
        LLAMA_FTYPE_MOSTLY_Q2_K,
        " 2.67G, +0.8698 ppl @ 7B - smallest, extreme quality loss - not recommended",
    },
    {
        "Q3_K",
        LLAMA_FTYPE_MOSTLY_Q3_K_M,
        "alias for Q3_K_M"
    },
    {
        "Q3_K_S",
        LLAMA_FTYPE_MOSTLY_Q3_K_S,
        " 2.75G, +0.5505 ppl @ 7B - very small, very high quality loss",
    },
    {
        "Q3_K_M",
        LLAMA_FTYPE_MOSTLY_Q3_K_M,
        " 3.06G, +0.2437 ppl @ 7B - very small, very high quality loss",
    },
    {
        "Q3_K_L",
        LLAMA_FTYPE_MOSTLY_Q3_K_L,
        " 3.35G, +0.1803 ppl @ 7B - small, substantial quality loss",
    },
    {
        "Q4_K",
        LLAMA_FTYPE_MOSTLY_Q4_K_M,
        "alias for Q4_K_M",
    },
    {
        "Q4_K_S",
        LLAMA_FTYPE_MOSTLY_Q4_K_S,
        " 3.56G, +0.1149 ppl @ 7B - small, significant quality loss",
    },
    {
        "Q4_K_M",
        LLAMA_FTYPE_MOSTLY_Q4_K_M,
        " 3.80G, +0.0535 ppl @ 7B - medium, balanced quality - *recommended*",
    },
    {
        "Q5_K",
        LLAMA_FTYPE_MOSTLY_Q5_K_M,
        "alias for Q5_K_M",
    },
    {
        "Q5_K_S",
        LLAMA_FTYPE_MOSTLY_Q5_K_S,
        " 4.33G, +0.0353 ppl @ 7B - large, low quality loss - *recommended*",
    },
    {
        "Q5_K_M",
        LLAMA_FTYPE_MOSTLY_Q5_K_M,
        " 4.45G, +0.0142 ppl @ 7B - large, very low quality loss - *recommended*",
    },
    {
        "Q6_K",
        LLAMA_FTYPE_MOSTLY_Q6_K,
        " 5.15G, +0.0044 ppl @ 7B - very large, extremely low quality loss",
    },
#endif
    {
        "Q8_0",
        LLAMA_FTYPE_MOSTLY_Q8_0,
        " 6.70G, +0.0004 ppl @ 7B - very large, extremely low quality loss - not recommended",
    },
    {
        "F16",
        LLAMA_FTYPE_MOSTLY_F16,
        "13.00G              @ 7B - extremely large, virtually no quality loss - not recommended",
    },
    {
        "F32",
        LLAMA_FTYPE_ALL_F32,
        "26.00G              @ 7B - absolutely huge, lossless - not recommended",
    },
};

bool try_parse_ftype(const std::string & ftype_str_in, llama_ftype & ftype, std::string & ftype_str_out) {
    std::string ftype_str;

    for (auto ch : ftype_str_in) {
        ftype_str.push_back(std::toupper(ch));
    }
    for (auto & it : QUANT_OPTIONS) {
        if (it.name == ftype_str) {
            ftype = it.ftype;
            ftype_str_out = it.name;
            return true;
        }
    }
    try {
        int ftype_int = std::stoi(ftype_str);
        for (auto & it : QUANT_OPTIONS) {
            if (it.ftype == ftype_int) {
                ftype = it.ftype;
                ftype_str_out = it.name;
                return true;
            }
        }
    }
    catch (...) {
        // stoi failed
    }
    return false;
}

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

//  if (!llama_model_quantize(fname_inp.c_str(), fname_out.c_str(), ftype, outError)) {
//    return NO;
//  }

//  return YES;

  return NO;
}

@end
