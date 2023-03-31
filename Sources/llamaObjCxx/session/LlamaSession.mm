//
//  LlamaSession.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 24/03/2023.
//

#import "LlamaSession.h"

#import "LlamaPredictOperation.hh"
#import "LlamaPredictionEvent.h"
#import "LlamaPredictionPayload.h"
#import "LlamaSetupOperation.hh"
#import "LlamaSessionConcretePredictionHandle.h"
#import "LlamaSessionParams.h"

typedef NS_ENUM(NSUInteger, LlamaSessionState) {
  LlamaSessionStateNone = 0,
  LlamaSessionStateLoadingModel,
  LlamaSessionStateReadyToPredict,
  LlamaSessionStatePredicting,
  LlamaSessionStateFailed
};

BOOL IsModelLoaded(LlamaSessionState state)
{
  switch (state) {
    case LlamaSessionStateNone:
    case LlamaSessionStateLoadingModel:
      return false;
    case LlamaSessionStateReadyToPredict:
    case LlamaSessionStatePredicting:
      return true;
    case LlamaSessionStateFailed:
      return false;
    default:
      return false;
  }
}

@interface _LlamaSession () <LlamaSetupOperationDelegate>
@end

@implementation _LlamaSession {
  _LlamaSessionParams *_params;
  NSOperationQueue *_operationQueue;

  NSMutableArray<LlamaPredictionPayload *> *_queuedPredictions;
  NSMutableArray<_LlamaSessionConcretePredictionHandle *> *_predictionHandles;

  LlamaSessionState _state;
  LlamaContext *_context;
}

- (instancetype)initWithParams:(_LlamaSessionParams *)params
                      delegate:(id<_LlamaSessionDelegate>)delegate
{
  if ((self = [super init])) {
    _params = params;
    _delegate = delegate;

    _operationQueue = [[NSOperationQueue alloc] init];
    _operationQueue.qualityOfService = NSQualityOfServiceUserInitiated;

    _predictionHandles = [[NSMutableArray alloc] init];

    _state = LlamaSessionStateNone;
    _queuedPredictions = [[NSMutableArray alloc] init];
  }
  return self;
}

#pragma mark - Loading

- (BOOL)_needsModelLoad
{
  if (_context != nil) {
    return false;
  }

  switch (_state) {
    case LlamaSessionStateNone:
      return true;
    case LlamaSessionStateLoadingModel:
    case LlamaSessionStateReadyToPredict:
    case LlamaSessionStateFailed:
      return false;
    default:
      return false;
  }
}

- (void)loadModelIfNeeded
{
  if (![self _needsModelLoad]) {
    return;
  }

  _state = LlamaSessionStateLoadingModel;
  [_delegate didStartLoadingModelInSession:self];

  LlamaSetupOperation *setupOperation = [[LlamaSetupOperation alloc] initWithParams:_params delegate:self];
  [_operationQueue addOperation:setupOperation];
}

#pragma mark - Predicting

- (id<_LlamaSessionPredictionHandle>)runPredictionWithPrompt:(NSString*)prompt
                                                startHandler:(void(^)(void))startHandler
                                                tokenHandler:(void(^)(NSString*))tokenHandler
                                           completionHandler:(void(^)(void))completionHandler
                                               cancelHandler:(void(^)(void))cancelHandler
                                              failureHandler:(void(^)(NSError*))failureHandler
                                                handlerQueue:(dispatch_queue_t)handlerQueue
{
  NSString *identifier = [[NSUUID UUID] UUIDString];
  LlamaPredictionPayload *payload = [[LlamaPredictionPayload alloc] initWithIdentifier:identifier
                                                                                prompt:prompt
                                                                          startHandler:startHandler
                                                                          tokenHandler:tokenHandler
                                                                     completionHandler:completionHandler
                                                                         cancelHandler:cancelHandler
                                                                        failureHandler:failureHandler
                                                                          handlerQueue:handlerQueue];

  if (!IsModelLoaded(_state)) {
    [_queuedPredictions addObject:payload];
    [self loadModelIfNeeded];
  } else {
    [self _runPredictionWithPayload:payload];
  }

  return [[_LlamaSessionConcretePredictionHandle alloc] initWithCancelHandler:^{
    // ensure to coordinate state on the main thread.
    dispatch_async(dispatch_get_main_queue(), ^{
      [self _cancelPredictionWithIdentifier:identifier];
    });
  }];
}

- (void)_runPredictionWithPayload:(LlamaPredictionPayload *)payload
{
  if (_context == nil) {
    return;
  }

  LlamaPredictOperationEventHandler operationEventHandler = ^(_LlamaPredictionEvent *event) {
    [event matchStarted:^{
      self->_state = LlamaSessionStatePredicting;
      [self->_delegate didStartPredictingInSession:self];
      if (payload.startHandler != NULL) {
        dispatch_async(payload.handlerQueue, ^{
          payload.startHandler();
        });
      }
    } outputToken:^(NSString *_Nonnull token) {
      if (payload.tokenHandler != NULL) {
        dispatch_async(payload.handlerQueue, ^{
          payload.tokenHandler(token);
        });
      }
    } completed:^{
      self->_state = LlamaSessionStateReadyToPredict;
      if (payload.completionHandler != NULL) {
        dispatch_async(payload.handlerQueue, ^{
          payload.completionHandler();
        });
      }
    } cancelled:^{
      self->_state = LlamaSessionStateReadyToPredict;
      if (payload.cancelHandler != NULL) {
        dispatch_async(payload.handlerQueue, ^{
          payload.cancelHandler();
        });
      }
    } failed:^(NSError * _Nonnull error) {
      self->_state = LlamaSessionStateFailed;
      if (payload.failureHandler != NULL) {
        dispatch_async(payload.handlerQueue, ^{
          payload.failureHandler(error);
        });
      }
    }];
  };

  LlamaPredictOperation *predictOperation = [[LlamaPredictOperation alloc] initWithIdentifier:payload.identifier
                                                                                      context:_context
                                                                                       prompt:payload.prompt
                                                                                 eventHandler:operationEventHandler
                                                                            eventHandlerQueue:dispatch_get_main_queue()];

  [_operationQueue addOperation:predictOperation];
}

#pragma mark - Cancellation

- (void)_cancelPredictionWithIdentifier:(NSString *)identifier
{
  NSAssert([NSThread isMainThread], @"Cancellation should be coordinated on the main thread");

  if ([self _cancelQueuedPredictionWithIdentifier:identifier]) {
    return;
  }

  // Check to see if it's in-flight
  for (NSOperation *operation in _operationQueue.operations) {
    if (![operation isKindOfClass:[LlamaPredictOperation class]]) {
      continue;
    }

    LlamaPredictOperation *predictOperation = (LlamaPredictOperation *)operation;
    if ([predictOperation.identifier isEqualToString:identifier] && predictOperation.isExecuting) {
      [predictOperation cancel];
      break;
    }
  }
}

- (BOOL)_cancelQueuedPredictionWithIdentifier:(NSString *)identifier
{

  // Check to see if it's queued
  LlamaPredictionPayload *payloadToRemove = nil;
  for (LlamaPredictionPayload *payload in _queuedPredictions) {
    if ([payload.identifier isEqualToString:identifier]) {
      payloadToRemove = payload;
      break;
    }
  }

  if (!payloadToRemove) {
    return NO;
  }

  [_queuedPredictions removeObject:payloadToRemove];
  return YES;
}

#pragma mark - LlamaSetupOperationDelegate

- (void)setupOperation:(nonnull LlamaSetupOperation *)operation didFailWithError:(nonnull NSError *)error
{
  _state = LlamaSessionStateFailed;
  [_delegate session:self didMoveToErrorStateWithError:error];
}

- (void)setupOperation:(nonnull LlamaSetupOperation *)operation didSucceedWithContext:(nonnull LlamaContext *)context
{
  _context = context;
  _state = LlamaSessionStateReadyToPredict;
  [_delegate didLoadModelInSession:self];

  for (LlamaPredictionPayload *payload in _queuedPredictions) {
    [self _runPredictionWithPayload:payload];
  }
  [_queuedPredictions removeAllObjects];
}

@end
