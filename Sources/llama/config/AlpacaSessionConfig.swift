//
//  AlpacaSessionConfig.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

public final class AlpacaSessionConfig: SessionConfig, ObjCxxParamsBuilder {
  // Based on values in https://github.com/ggerganov/llama.cpp/blob/107980d/examples/alpaca.sh
  public static var `default`: Self {
    return Self.init(
      numTokens: 512,
      hyperparameters: Hyperparameters(
        contextSize: 2048,
        batchSize: 256,
        topK: 10000,
        temperature: 0.2,
        repeatPenalty: 1
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
