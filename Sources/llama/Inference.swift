//
//  Inference.swift
//  llama
//
//  Created by Alex Rozanski on 24/03/2023.
//

import Foundation

public class Inference {
  public struct Config {
    public let numThreads: UInt

    public init(numThreads: UInt) {
      self.numThreads = numThreads
    }

    public static let `default` = Config(numThreads: UInt(ProcessInfo().activeProcessorCount))
  }

  public let config: Config

  public init(config: Config) {
    self.config = config
  }

  // MARK: - Sessions
  public func makeLlamaSession(
    with modelURL: URL,
    config: LlamaSessionConfig,
    stateChangeHandler: Session.StateChangeHandler?
  ) -> Session {
    return BridgedSession(
      modelURL: modelURL,
      configBuilder: LlamaSessionConfigBuilder(sessionConfig: config, inferenceConfig: self.config),
      stateChangeHandler: stateChangeHandler
    )
  }

  public func makeAlpacaSession(
    with modelURL: URL,
    config: AlpacaSessionConfig,
    stateChangeHandler: Session.StateChangeHandler?
  ) -> Session {
    return BridgedSession(
      modelURL: modelURL,
      configBuilder: AlpacaSessionConfigBuilder(sessionConfig: config, inferenceConfig: self.config),
      stateChangeHandler: stateChangeHandler
    )
  }
}
