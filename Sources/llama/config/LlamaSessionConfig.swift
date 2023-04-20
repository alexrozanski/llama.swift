//
//  LlamaSessionConfig.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

public final class LlamaSessionConfig: SessionConfig, ObjCxxParamsBuilder {
  public static var `defaults`: LlamaSessionConfig {
    return configurableDefaults.build()
  }

  public static var configurableDefaults: SessionConfigBuilder<LlamaSessionConfig> {
    return SessionConfigBuilder(defaults: defaultSessionConfig)
      .withNumTokens(128)
  }

  func build(for modelURL: URL) -> _LlamaSessionParams {
    return SessionConfigParamsBuilder(sessionConfig: self, mode: .regular).build(for: modelURL)
  }
}
