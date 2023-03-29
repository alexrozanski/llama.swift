//
//  AlpacaSessionConfigBuilder.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

public class AlpacaSessionConfig: GeneralSessionConfig {}

class AlpacaSessionConfigBuilder: ObjCxxConfigBuilder {
  let sessionConfig: AlpacaSessionConfig
  let inferenceConfig: Inference.Config

  init(sessionConfig: AlpacaSessionConfig, inferenceConfig: Inference.Config) {
    self.sessionConfig = sessionConfig
    self.inferenceConfig = inferenceConfig
  }

  func build() -> _LlamaSessionConfig {
    let _config = _LlamaSessionConfig(mode: .instructional)
    _config.numberOfThreads = inferenceConfig.numThreads
    _config.numberOfTokens = sessionConfig.numTokens
    _config.reversePrompt = sessionConfig.reversePrompt
    _config.seed = sessionConfig.seed ?? 0
    return _config
  }
}
