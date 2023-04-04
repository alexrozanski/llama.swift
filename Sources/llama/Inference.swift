//
//  Inference.swift
//  llama
//
//  Created by Alex Rozanski on 24/03/2023.
//

import Foundation
import llamaObjCxx

public class Inference {
  public struct Config {
    public let numThreads: UInt

    public init(numThreads: UInt) {
      self.numThreads = numThreads
    }

    public static let `default`: Config = {
      let processorCount = UInt(ProcessInfo().activeProcessorCount)
      // Account for main thread and worker thread. Specifying all active processors seems to introduce a lot of contention.
      let maxAvailableProcessors = processorCount - 2
      // Experimentally 6 also seems like a pretty good number.
      let numThreads = min(maxAvailableProcessors, 6)
      return Config(numThreads: numThreads)
    }()
  }

  public let config: Config

  public init(config: Config) {
    self.config = config
  }

  // MARK: - Models

  public static func getModelType(forFileAt fileURL: URL) throws -> ModelType {
    return try BridgedSession.getModelType(forFileAt: fileURL)
  }

  public static func validateModel(fileURL: URL) throws {
    try BridgedSession.validateModel(fileURL: fileURL)
  }

  // MARK: - Sessions
  public func makeLlamaSession(
    with modelURL: URL,
    config: LlamaSessionConfig
  ) -> Session {
    return BridgedSession(
      paramsBuilder: LlamaSessionParamsBuilder(modelURL: modelURL, sessionConfig: config, inferenceConfig: self.config)
    )
  }

  public func makeAlpacaSession(
    with modelURL: URL,
    config: AlpacaSessionConfig
  ) -> Session {
    return BridgedSession(
      paramsBuilder: AlpacaSessionParamsBuilder(modelURL: modelURL, sessionConfig: config, inferenceConfig: self.config)
    )
  }

  public func makeGPT4AllSession(
    with modelURL: URL,
    config: GPT4AllSessionConfig
  ) -> Session {
    return BridgedSession(
      paramsBuilder: GPT4AllSessionParamsBuilder(modelURL: modelURL, sessionConfig: config, inferenceConfig: self.config)
    )
  }
}
