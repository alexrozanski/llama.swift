//
//  LlamaSessionConfig.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

public final class LlamaSessionConfig: SessionConfig, ObjCxxParamsBuilder {
  public static var `default`: Self {
    return self.mergeIntoDefaults(
      from: Self.init(numTokens: 128, hyperparameters: Hyperparameters())
    )
  }

  func build(for modelURL: URL) -> _LlamaSessionParams {
    return SessionConfigBuilder(sessionConfig: self, mode: .regular).build(for: modelURL)
  }
}
