//
//  LlamaSessionContext+Internal.h
//  llama
//
//  Created by Alex Rozanski on 02/04/2023.
//

#import "LlamaSessionContext.h"

@interface _LlamaSessionContextToken ()
- (nonnull instancetype)initWithToken:(int)token string:(NSString *__nonnull)string;
@end

@interface _LlamaSessionContext ()
- (nonnull instancetype)initWithTokens:(NSArray<_LlamaSessionContextToken *> *__nullable)tokens;
@end
