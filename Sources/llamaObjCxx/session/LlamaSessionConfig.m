//
//  LlamaSessionConfig.m
//  llama
//
//  Created by Alex Rozanski on 13/03/2023.
//

#import "LlamaSessionConfig.h"

@implementation _LlamaSessionConfig

@synthesize mode = _mode;
@synthesize numberOfThreads = _numberOfThreads;
@synthesize numberOfTokens = _numberOfTokens;
@synthesize reversePrompt = _reversePrompt;
@synthesize seed = _seed;

- (instancetype)initWithMode:(_LlamaSessionMode)mode
{
  if ((self = [super init])) {
    _mode = mode;
  }

  return self;
}

@end
