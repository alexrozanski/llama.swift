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

// Wraps llama_ftype
typedef NS_ENUM(NSUInteger, _LlamaModelFileType) {
  _LlamaModelFileTypeUnknown = 0,
  _LlamaModelFileTypeAllF32 = 1,
  _LlamaModelFileTypeMostlyF16 = 2,
  _LlamaModelFileTypeMostlyQ4_0 = 3,
  _LlamaModelFileTypeMostlyQ4_1 = 4,
  _LlamaModelFileTypeMostlyQ4_1SomeF16 = 5
};

@interface _LlamaModelUtils : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

// MARK: - Models

+ (BOOL)loadModelTypeForFileAtURL:(NSURL *)fileURL
                     outModelType:(_LlamaModelType *)outModelType
                         outError:(NSError **)outError;

+ (BOOL)quantizeModelWithSourceFileURL:(NSURL *)fileURL
                           destFileURL:(NSURL *)destFileURL
                              fileType:(_LlamaModelFileType)quantizationType
                              outError:(NSError **)outError;


@end

NS_ASSUME_NONNULL_END
