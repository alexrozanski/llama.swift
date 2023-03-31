//
//  LlamaPredictionPayload.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 25/03/2023.
//

#import "LlamaPredictionPayload.h"

@implementation LlamaPredictionPayload

- (instancetype)initWithIdentifier:(NSString *)identifier
                            prompt:(NSString *)prompt
                      tokenHandler:(void(^)(NSString*))tokenHandler
                 completionHandler:(void(^)(void))completionHandler
                    failureHandler:(void(^)(NSError*))failureHandler
                      handlerQueue:(dispatch_queue_t)handlerQueue
{
  if ((self = [super init])) {
    _identifier = [identifier copy];
    _prompt = [prompt copy];
    _tokenHandler = [tokenHandler copy];
    _completionHandler = [completionHandler copy];
    _failureHandler = [failureHandler copy];
    _handlerQueue = handlerQueue;
  }

  return self;
}

@end
