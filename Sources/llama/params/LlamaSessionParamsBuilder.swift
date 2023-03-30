//
//  LlamaSessionParamsBuilder.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

class LlamaSessionParamsBuilder: ObjCxxParamsBuilder {
  let sessionConfig: LlamaSessionConfig
  let inferenceConfig: Inference.Config

  init(sessionConfig: LlamaSessionConfig, inferenceConfig: Inference.Config) {
    self.sessionConfig = sessionConfig
    self.inferenceConfig = inferenceConfig
  }

  func build() -> _LlamaSessionParams {
    let params = _LlamaSessionParams(mode: .regular)
    params.numberOfThreads = inferenceConfig.numThreads
    params.numberOfTokens = sessionConfig.numTokens
    params.antiprompts = [sessionConfig.reversePrompt].compactMap { $0 }
    params.seed = sessionConfig.seed ?? 0
    return params
  }
}
