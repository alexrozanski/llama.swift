//
//  AlpacaSessionParamsBuilder.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

class AlpacaSessionParamsBuilder: ObjCxxParamsBuilder {
  let sessionConfig: AlpacaSessionConfig
  let inferenceConfig: Inference.Config

  init(sessionConfig: AlpacaSessionConfig, inferenceConfig: Inference.Config) {
    self.sessionConfig = sessionConfig
    self.inferenceConfig = inferenceConfig
  }

  func build() -> _LlamaSessionParams {
    let params = _LlamaSessionParams(mode: .instructional)
    params.numberOfThreads = inferenceConfig.numThreads

    params.initialPrompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    params.promptPrefix = "\n\n### Instruction:\n\n"
    params.promptSuffix = "\n\n### Response:\n\n"
    params.antiprompts = ["\n\n### Response:\n\n", sessionConfig.reversePrompt].compactMap { $0 }

    params.numberOfTokens = sessionConfig.numTokens
    params.seed = sessionConfig.seed ?? 0

    return params
  }
}
