//
//  LlamaEvent.mm
//  llama
//
//  Created by Alex Rozanski on 14/03/2023.
//

#include "LlamaEvent.h"

typedef NS_ENUM(NSUInteger, LlamaEventType) {
  LlamaEventTypeNone = 0,
  LlamaEventTypeStartedLoadingModel,
  LlamaEventTypeFinishedLoadingModel,
  LlamaEventTypeStartedGeneratingOutput,
  LlamaEventTypeOutputToken,
  LlamaEventTypeCompleted,
  LlamaEventTypeFailed,
};

typedef struct LlamaEventData {
  NSString *outputToken_token;
  NSError *failed_error;
} LlamaEventData;

@interface _LlamaEvent () {
  LlamaEventType _eventType;
  LlamaEventData _data;
}

- (instancetype)initWithEventType:(LlamaEventType)eventType data:(LlamaEventData)data;

@end

@implementation _LlamaEvent

- (instancetype)initWithEventType:(LlamaEventType)eventType data:(LlamaEventData)data
{
  if ((self = [super init])) {
    _eventType = eventType;
    _data = data;
  }

  return self;
}

+ (instancetype)startedLoadingModel
{
  LlamaEventData data;
  _LlamaEvent *event = [[_LlamaEvent alloc] initWithEventType:LlamaEventTypeStartedLoadingModel data:{}];
  return event;
}

+ (instancetype)finishedLoadingModel
{
  LlamaEventData data;
  _LlamaEvent *event = [[_LlamaEvent alloc] initWithEventType:LlamaEventTypeFinishedLoadingModel data:{}];
  return event;
}

+ (instancetype)startedGeneratingOutput
{
  LlamaEventData data;
  _LlamaEvent *event = [[_LlamaEvent alloc] initWithEventType:LlamaEventTypeStartedGeneratingOutput data:{}];
  return event;
}

+ (instancetype)outputTokenWithToken:(nonnull NSString *)token
{
  _LlamaEvent *event = [[_LlamaEvent alloc] initWithEventType:LlamaEventTypeOutputToken data:{ .outputToken_token = [token copy] }];
  return event;
}

+ (instancetype)completed
{
  _LlamaEvent *event = [[_LlamaEvent alloc] initWithEventType:LlamaEventTypeCompleted data:{}];
  return event;
}

+ (instancetype)failedWithError:(nonnull NSError *)error
{
  _LlamaEvent *event = [[_LlamaEvent alloc] initWithEventType:LlamaEventTypeFailed data:{ .failed_error = [error copy] }];
  return event;
}

- (void)matchWithStartedLoadingModel:(void (^)(void))startedLoadingModel
                finishedLoadingModel:(void (^)(void))finishedLoadingModel
             startedGeneratingOutput:(void (^)(void))startedGeneratingOutput
                         outputToken:(void (^)(NSString *token))outputToken
                           completed:(void (^)(void))completed
                              failed:(void (^)(NSError *error))failed
{
  switch (_eventType) {
    case LlamaEventTypeNone:
      break;
    case LlamaEventTypeStartedLoadingModel:
      startedLoadingModel();
      break;
    case LlamaEventTypeFinishedLoadingModel:
      finishedLoadingModel();
      break;
    case LlamaEventTypeStartedGeneratingOutput:
      startedGeneratingOutput();
      break;
    case LlamaEventTypeOutputToken:
      outputToken(_data.outputToken_token);
      break;
    case LlamaEventTypeCompleted:
      completed();
      break;
    case LlamaEventTypeFailed:
      failed(_data.failed_error);
      break;
  }
}

@end
