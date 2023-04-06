//
//  LlamaModelUtils.h
//  llamaObjCxx
//
//  Created by Alex Rozanski on 05/04/2023.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, _LlamaModelType) {
  _LlamaModelTypeUnknown = 0,
  _LlamaModelType7B,
  _LlamaModelType13B,
  _LlamaModelType30B,
  _LlamaModelType65B
};

@interface _LlamaModelUtils : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

// MARK: - Models

+ (BOOL)loadModelTypeForFileAtURL:(NSURL *)fileURL
                     outModelType:(_LlamaModelType *)outModelType
                         outError:(NSError **)outError;

@end

NS_ASSUME_NONNULL_END
