//
//  LlamaSessionContext.h
//  llama
//
//  Created by Alex Rozanski on 02/04/2023.
//

#import <Foundation/Foundation.h>

@interface _LlamaSessionContext : NSObject

@property (nonatomic, readonly, copy, nullable) NSString *contextString;
@property (nonatomic, readonly, copy, nullable) NSArray<NSNumber *> *tokens;

@end
