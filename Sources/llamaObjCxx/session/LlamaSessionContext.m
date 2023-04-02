//
//  LlamaSessionContext.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 02/04/2023.
//

#import "LlamaSessionContext.h"

@implementation _LlamaSessionContext

- (instancetype)initWithContextString:(NSString *__nullable)contextString tokens:(NSArray<NSNumber *> *__nullable)tokens
{
  if ((self = [super init])) {
    _contextString = [contextString copy];

    // Deep copy this just to make sure this is really immutable.
    NSMutableArray *deepCopiedTokens = [[NSMutableArray alloc] initWithCapacity:tokens.count];
    for (NSNumber *token in tokens) {
      [deepCopiedTokens addObject:[token copy]];
    }
    _tokens = [deepCopiedTokens copy];
  }
  return self;
}

@end
