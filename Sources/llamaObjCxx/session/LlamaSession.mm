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

- (void)runPredictionWithPrompt:(NSString*)prompt
                   tokenHandler:(void(^)(NSString*))tokenHandler
              completionHandler:(void(^)(void))completionHandler
                 failureHandler:(void(^)(NSError*))failureHandler
{
  LlamaPredictionPayload *payload = [[LlamaPredictionPayload alloc] initWithPrompt:prompt
                                                                      tokenHandler:tokenHandler
                                                                 completionHandler:completionHandler
                                                                    failureHandler:failureHandler];

  if (!IsModelLoaded(_state)) {
    [_queuedPredictions addObject:payload];
    [self loadModelIfNeeded];
  } else {
    [self _runPredictionWithPayload:payload];
  }
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
    } outputToken:^(NSString *_Nonnull token) {
      if (payload.tokenHandler != NULL) {
        payload.tokenHandler(token);
      }
    } completed:^{
      if (payload.completionHandler != NULL) {
        payload.completionHandler();
      }
    } failed:^(NSError * _Nonnull error) {
      if (payload.failureHandler != NULL) {
        payload.failureHandler(error);
      }
    }];
  };

  LlamaPredictOperation *predictOperation = [[LlamaPredictOperation alloc] initWithContext:_context
                                                                                    prompt:payload.prompt
                                                                              eventHandler:operationEventHandler
                                                                         eventHandlerQueue:dispatch_get_main_queue()];

  [_operationQueue addOperation:predictOperation];
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
