//
//  LlamaSessionConfigBuilder.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

public class LlamaSessionConfig: GeneralSessionConfig {}

class LlamaSessionConfigBuilder: ObjCxxConfigBuilder {
  let sessionConfig: LlamaSessionConfig
  let inferenceConfig: Inference.Config

  init(sessionConfig: LlamaSessionConfig, inferenceConfig: Inference.Config) {
    self.sessionConfig = sessionConfig
    self.inferenceConfig = inferenceConfig
  }

  func build() -> _LlamaSessionConfig {
    let _config = _LlamaSessionConfig(mode: .regular)
    _config.numberOfThreads = inferenceConfig.numThreads
    _config.numberOfTokens = sessionConfig.numTokens
    _config.reversePrompt = sessionConfig.reversePrompt
    _config.seed = sessionConfig.seed ?? 0
    return _config
  }
}
