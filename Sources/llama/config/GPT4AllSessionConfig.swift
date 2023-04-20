//
//  GPT4AllSessionConfig.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

public final class GPT4AllSessionConfig: SessionConfig, ObjCxxParamsBuilder {
  public static var `defaults`: GPT4AllSessionConfig {
    return configurableDefaults.build()
  }

  // Based on config in https://github.com/ggerganov/llama.cpp/blob/437e77855a54e69c86fe03bc501f63d9a3fddb0e/examples/gpt4all.sh#L10
  public static var configurableDefaults: SessionConfigBuilder<GPT4AllSessionConfig> {
    return SessionConfigBuilder(defaults: defaultSessionConfig)
      .withNumTokens(128)
      .withHyperparameters { hyperparameters in
        hyperparameters
          .withContextSize(2048)
          .withBatchSize(8)
          .withLastNTokensToPenalize(64)
          .withTopK(40)
          .withTopP(0.95)
          .withTemperature(0.1)
          .withRepeatPenalty(1.3)
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
