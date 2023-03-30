//
//  LlamaSessionParams.m
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import "LlamaSessionParams.h"

@implementation _LlamaSessionParams

@synthesize mode = _mode;

@synthesize initialPrompt = _initialPrompt;
@synthesize promptPrefix = _promptPrefix;
@synthesize promptSuffix = _promptSuffix;

@synthesize numberOfThreads = _numberOfThreads;
@synthesize numberOfTokens = _numberOfTokens;
@synthesize antiprompts = _antiprompts;
@synthesize seed = _seed;

- (instancetype)initWithMode:(_LlamaSessionMode)mode
{
  if ((self = [super init])) {
    _mode = mode;
  }

  return self;
}

@end
