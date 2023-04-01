//
//  AlpacaSessionParamsBuilder.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

class AlpacaSessionParamsBuilder: ObjCxxParamsBuilder {
  let modelURL: URL
  let sessionConfig: AlpacaSessionConfig
  let inferenceConfig: Inference.Config

  init(modelURL: URL, sessionConfig: AlpacaSessionConfig, inferenceConfig: Inference.Config) {
    self.modelURL = modelURL
    self.sessionConfig = sessionConfig
    self.inferenceConfig = inferenceConfig
  }

  func build() -> _LlamaSessionParams {
    let params = _LlamaSessionParams.defaultParams(withModelPath: modelURL.path, mode: .instructional)
    params.numberOfThreads = Int32(inferenceConfig.numThreads)

    params.initialPrompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    params.promptPrefix = "\n\n### Instruction:\n\n"
    params.promptSuffix = "\n\n### Response:\n\n"
    params.antiprompts = ["\n\n### Response:\n\n", sessionConfig.reversePrompt].compactMap { $0 }

    params.numberOfTokens = Int32(sessionConfig.numTokens)
    params.seed = sessionConfig.seed ?? 0

    // matches config in https://github.com/ggerganov/llama.cpp/blob/a717cba8440b380f43cd3e2510862fc1ea3de9a2/examples/alpaca.sh#L10
    params.batchSize = 256
    params.topK = 10000
    params.temp = 0.2
    params.repeatPenalty = 1

    return params
  }
}
