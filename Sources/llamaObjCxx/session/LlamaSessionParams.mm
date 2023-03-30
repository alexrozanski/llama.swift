//
//  LlamaSessionParams.m
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import "LlamaSessionParams.h"

#include <thread>

@interface _LlamaSessionParams ()
- (instancetype)initWithModelPath:(NSString *)modelPath mode:(_LlamaSessionMode)mode;
@end

@implementation _LlamaSessionParams

- (instancetype)initWithModelPath:(NSString *)modelPath mode:(_LlamaSessionMode)mode
{
  if ((self = [super init])) {
    _modelPath = [modelPath copy];
    _mode = mode;
  }

  return self;
}

+ (instancetype)defaultParamsWithModelPath:(NSString *)modelPath mode:(_LlamaSessionMode)mode;
{
  _LlamaSessionParams *params = [[self alloc] initWithModelPath:modelPath mode:mode];

  params.seed = -1;
  params.numberOfThreads = std::min(4, (int32_t) std::thread::hardware_concurrency());
  params.numberOfTokens = 128;
  params.numberOfTokensToPenalize = 64;
  params.numberOfParts = -1;
  params.contextSize = 512;
  params.batchSize = 8;
  params.numberOfTokensToKeepFromInitialPrompt = 0;

  params.topK = 40;
  params.topP = 0.95f;
  params.temp = 0.80f;
  params.repeatPenalty = 1.10f;

  params.useF16Memory = YES;

  return params;
}

#pragma mark - Convenience properties

- (BOOL)isInstructional
{
  switch (_mode) {
    case _LlamaSessionModeRegular:
      return NO;
    case _LlamaSessionModeInstructional:
      return YES;
  }
}

@end
