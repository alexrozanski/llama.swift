//
//  Inference.swift
//  llama
//
//  Created by Alex Rozanski on 24/03/2023.
//

import Foundation
import llamaObjCxx

public class SessionManager {
  public init() {}  

  // MARK: - Sessions
  public func makeLlamaSession(
    with modelURL: URL,
    config: LlamaSessionConfig
  ) -> Session {
    return BridgedSession(
      modelURL: modelURL,
      paramsBuilder: config
    )
  }

  public func makeAlpacaSession(
    with modelURL: URL,
    config: AlpacaSessionConfig
  ) -> Session {
    return BridgedSession(
      modelURL: modelURL,
      paramsBuilder: config
    )
  }

  public func makeGPT4AllSession(
    with modelURL: URL,
    config: GPT4AllSessionConfig
  ) -> Session {
    return BridgedSession(
      modelURL: modelURL,
      paramsBuilder: config
    )
  }
}
