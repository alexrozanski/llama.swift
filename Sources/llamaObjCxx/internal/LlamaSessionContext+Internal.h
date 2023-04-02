//
//  LlamaSessionContext+Internal.h
//  llama
//
//  Created by Alex Rozanski on 02/04/2023.
//

#import "LlamaSessionContext.h"

@interface _LlamaSessionContext ()
- (instancetype)initWithContextString:(NSString *__nullable)contextString tokens:(NSArray<NSNumber *> *__nullable)tokens;
@end
