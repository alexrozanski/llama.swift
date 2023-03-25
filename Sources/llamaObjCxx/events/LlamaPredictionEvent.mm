//
//  LlamaPredictionEvent.cpp
//  llamaObjCxx
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import "LlamaPredictionEvent.h"

typedef NS_ENUM(NSUInteger, LlamaPredictionEventType) {
  LlamaPredictionEventTypeNone = 0,
  LlamaPredictionEventTypeStarted,
  LlamaPredictionEventTypeOutputToken,
  LlamaPredictionEventTypeCompleted,
  LlamaPredictionEventTypeFailed,
};

typedef struct LlamaPredictionEventData {
  NSString *outputToken_token;
  NSError *failed_error;
} LlamaPredictionEventData;

@interface _LlamaPredictionEvent () {
  LlamaPredictionEventType _eventType;
  LlamaPredictionEventData _data;
}

- (instancetype)initWithEventType:(LlamaPredictionEventType)eventType data:(LlamaPredictionEventData)data;

@end

@implementation _LlamaPredictionEvent

- (instancetype)initWithEventType:(LlamaPredictionEventType)eventType data:(LlamaPredictionEventData)data
{
  if ((self = [super init])) {
    _eventType = eventType;
    _data = data;
  }

  return self;
}

+ (instancetype)started
{
  return [[_LlamaPredictionEvent alloc] initWithEventType:LlamaPredictionEventTypeStarted
                                                     data:{}];
}

+ (instancetype)outputTokenWithToken:(nonnull NSString *)token
{
  return [[_LlamaPredictionEvent alloc] initWithEventType:LlamaPredictionEventTypeOutputToken
                                                     data:{ .outputToken_token = [token copy] }];
}

+ (instancetype)completed
{
  return [[_LlamaPredictionEvent alloc] initWithEventType:LlamaPredictionEventTypeCompleted
                                                     data:{}];
}

+ (instancetype)failedWithError:(nonnull NSError *)error
{
  return [[_LlamaPredictionEvent alloc] initWithEventType:LlamaPredictionEventTypeFailed
                                                     data:{ .failed_error = [error copy] }];
}

- (void)matchStarted:(void (^)(void))started
         outputToken:(void (^)(NSString *token))outputToken
           completed:(void (^)(void))completed
              failed:(void (^)(NSError *error))failed
{
  switch (_eventType) {
    case LlamaPredictionEventTypeNone:
      break;
    case LlamaPredictionEventTypeStarted:
      started();
      break;
    case LlamaPredictionEventTypeOutputToken:
      outputToken(_data.outputToken_token);
      break;
    case LlamaPredictionEventTypeCompleted:
      completed();
      break;
    case LlamaPredictionEventTypeFailed:
      failed(_data.failed_error);
      break;
  }
}

@end
