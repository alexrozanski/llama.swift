//
//  LlamaSessionParamsBuilder.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

class LlamaSessionParamsBuilder: ObjCxxParamsBuilder {
  let modelURL: URL
  let sessionConfig: LlamaSessionConfig

  init(modelURL: URL, sessionConfig: LlamaSessionConfig) {
    self.modelURL = modelURL
    self.sessionConfig = sessionConfig
  }

  func build() -> _LlamaSessionParams {
    let params = _LlamaSessionParams.defaultParams(withModelPath: modelURL.path, mode: .regular)
    params.numberOfThreads = Int32(sessionConfig.numThreads)
    params.numberOfTokens = Int32(sessionConfig.numTokens)
    params.antiprompts = [sessionConfig.reversePrompt].compactMap { $0 }
    params.seed = sessionConfig.seed ?? 0
    return params
  }
}
