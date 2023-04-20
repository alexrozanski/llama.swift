//
//  AlpacaSessionConfig.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

public final class AlpacaSessionConfig: SessionConfig, ObjCxxParamsBuilder {
  public static var `defaults`: AlpacaSessionConfig {
    return configurableDefaults.build()
  }

  // Based on values in https://github.com/ggerganov/llama.cpp/blob/107980d/examples/alpaca.sh
  public static var configurableDefaults: SessionConfigBuilder<AlpacaSessionConfig> {
    return SessionConfigBuilder(defaults: defaultSessionConfig)
      .withNumTokens(512)
      .withHyperparameters { hyperparameters in
        hyperparameters
          .withContextSize(2048)
          .withBatchSize(256)
          .withTopK(10000)
          .withTemperature(0.2)
          .withRepeatPenalty(1)
      }
  }

  func build(for modelURL: URL) -> _LlamaSessionParams {
    let params = SessionConfigParamsBuilder(sessionConfig: self, mode: .instructional).build(for: modelURL)

    params.initialPrompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    params.promptPrefix = "\n\n### Instruction:\n\n"
    params.promptSuffix = "\n\n### Response:\n\n"
    params.antiprompts = ["### Instruction:\n\n", reversePrompt].compactMap { $0 }

    return params
  }
}
