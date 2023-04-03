//
//  GPT4AllSessionParamsBuilder.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

class GPT4AllSessionParamsBuilder: ObjCxxParamsBuilder {
  let modelURL: URL
  let sessionConfig: GPT4AllSessionConfig
  let inferenceConfig: Inference.Config

  init(modelURL: URL, sessionConfig: GPT4AllSessionConfig, inferenceConfig: Inference.Config) {
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
    params.antiprompts = ["### Instruction:\n\n", sessionConfig.reversePrompt].compactMap { $0 }

    params.numberOfTokens = Int32(sessionConfig.numTokens)
    params.seed = sessionConfig.seed ?? 0

    // matches config in https://github.com/ggerganov/llama.cpp/blob/437e77855a54e69c86fe03bc501f63d9a3fddb0e/examples/gpt4all.sh#L10
    params.batchSize = 8
    params.contextSize = 2048
    params.numberOfTokensToPenalize = 64
    params.repeatPenalty = 1.3
    params.numberOfTokens = 128
    params.temp = 0.1
    params.topK = 40
    params.topP = 0.95

    return params
  }
}
