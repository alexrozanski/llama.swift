//
//  LlamaPredictionPayload.m
//  llamaObjCxx
//
//  Created by Alex Rozanski on 25/03/2023.
//

#import "LlamaPredictionPayload.h"

@implementation LlamaPredictionPayload

- (instancetype)initWithPrompt:(NSString *)prompt
                  tokenHandler:(void(^)(NSString*))tokenHandler
             completionHandler:(void(^)(void))completionHandler
                failureHandler:(void(^)(NSError*))failureHandler
{
  if ((self = [super init])) {
    _prompt = [prompt copy];
    _tokenHandler = [tokenHandler copy];
    _completionHandler = [completionHandler copy];
    _failureHandler = [failureHandler copy];
  }

  return self;
}

@end
