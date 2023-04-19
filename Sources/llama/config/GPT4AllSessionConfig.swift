//
//  GPT4AllSessionConfig.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

public final class GPT4AllSessionConfig: SessionConfig, ObjCxxParamsBuilder {
  // Based on config in https://github.com/ggerganov/llama.cpp/blob/437e77855a54e69c86fe03bc501f63d9a3fddb0e/examples/gpt4all.sh#L10
  public static var `default`: Self {
    return Self.init(
      numTokens: 128,
      hyperparameters: Hyperparameters(
        contextSize: 2048,
        batchSize: 8,
        lastNTokensToPenalize: 64,
        topK: 40,
        topP: 0.95,
        temperature: 0.1,
        repeatPenalty: 1.3
      )
    )
  }

  func build(for modelURL: URL) -> _LlamaSessionParams {
    let params = SessionConfigBuilder(sessionConfig: self, mode: .instructional).build(for: modelURL)

    params.initialPrompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    params.promptPrefix = "\n\n### Instruction:\n\n"
    params.promptSuffix = "\n\n### Response:\n\n"
    params.antiprompts = ["### Instruction:\n\n", reversePrompt].compactMap { $0 }

    return params
  }
}
