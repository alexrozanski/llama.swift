//
//  SessionConfig.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

public struct Hyperparameters {
  // The number of tokens to keep as context
  public fileprivate(set) var contextSize: UInt
  public fileprivate(set) var batchSize: UInt
  public fileprivate(set) var lastNTokensToPenalize: UInt
  public fileprivate(set) var topK: UInt
  // Should be between 0 and 1
  public fileprivate(set) var topP: Double
  public fileprivate(set) var temperature: Double
  public fileprivate(set) var repeatPenalty: Double

  public init(
    contextSize: UInt,
    batchSize: UInt,
    lastNTokensToPenalize: UInt,
    topK: UInt,
    topP: Double,
    temperature: Double,
    repeatPenalty: Double
  ) {
    self.contextSize = contextSize
    self.batchSize = batchSize
    self.lastNTokensToPenalize = lastNTokensToPenalize
    self.topK = topK
    self.topP = topP
    self.temperature = temperature
    self.repeatPenalty = repeatPenalty
  }
}

public class SessionConfig {
  // Seed for generation
  public private(set) var seed: Int32?

  // Number of threads to run prediction on.
  public private(set) var numThreads: UInt

  // Number of tokens to predict for each run.
  public private(set) var numTokens: UInt

  // Model configuration
  public private(set) var hyperparameters: Hyperparameters

  public let reversePrompt: String?

  required init(
    seed: Int32? = nil,
    numThreads: UInt,
    numTokens: UInt,
    hyperparameters: Hyperparameters,
    reversePrompt: String?
  ) {
    self.seed = seed
    self.numThreads = numThreads
    self.numTokens = numTokens
    self.hyperparameters = hyperparameters
    self.reversePrompt = reversePrompt
  }
}

// MARK: - Config Builders

public class HyperparametersBuilder {
  public private(set) var contextSize: UInt?
  public private(set) var batchSize: UInt?
  public private(set) var lastNTokensToPenalize: UInt?
  public private(set) var topK: UInt?
  public private(set) var topP: Double?
  public private(set) var temperature: Double?
  public private(set) var repeatPenalty: Double?

  private let defaults: Hyperparameters

  init(defaults: Hyperparameters) {
    self.defaults = defaults
  }

  public func withContextSize(_ contextSize: UInt?) -> Self {
    self.contextSize = contextSize
    return self
  }

  public func withBatchSize(_ batchSize: UInt?) -> Self {
    self.batchSize = batchSize
    return self
  }

  public func withLastNTokensToPenalize(_ lastNTokensToPenalize: UInt?) -> Self {
    self.lastNTokensToPenalize = lastNTokensToPenalize
    return self
  }

  public func withTopK(_ topK: UInt?) -> Self {
    self.topK = topK
    return self
  }

  public func withTopP(_ topP: Double?) -> Self {
    self.topP = topP
    return self
  }

  public func withTemperature(_ temperature: Double?) -> Self {
    self.temperature = temperature
    return self
  }

  public func withRepeatPenalty(_ repeatPenalty: Double?) -> Self {
    self.repeatPenalty = repeatPenalty
    return self
  }

  func build() -> Hyperparameters {
    return Hyperparameters(
      contextSize: contextSize ?? defaults.contextSize,
      batchSize: batchSize ?? defaults.batchSize,
      lastNTokensToPenalize: lastNTokensToPenalize ?? defaults.lastNTokensToPenalize,
      topK: topK ?? defaults.topK,
      topP: topP ?? defaults.topP,
      temperature: temperature ?? defaults.temperature,
      repeatPenalty: repeatPenalty ?? defaults.repeatPenalty
    )
  }
}

public class SessionConfigBuilder<T> where T: SessionConfig {
  public private(set) var seed: Int32??
  public private(set) var numThreads: UInt?
  public private(set) var numTokens: UInt?
  public private(set) var hyperparameters: HyperparametersBuilder
  public private(set) var reversePrompt: String??

  private let defaults: SessionConfig

  public init(defaults: SessionConfig) {
    self.hyperparameters = HyperparametersBuilder(defaults: defaults.hyperparameters)
    self.defaults = defaults
  }

  public func withSeed(_ seed: Int32?) -> Self {
    self.seed = seed
    return self
  }

  public func withNumThreads(_ numThreads: UInt?) -> Self {
    self.numThreads = numThreads
    return self
  }

  public func withNumTokens(_ numTokens: UInt?) -> Self {
    self.numTokens = numTokens
    return self
  }

  public func withHyperparameters(_ hyperParametersConfig: (HyperparametersBuilder) -> HyperparametersBuilder) -> Self {
    self.hyperparameters = hyperParametersConfig(hyperparameters)
    return self
  }

  public func withReversePrompt(_ reversePrompt: String??) -> Self {
    self.reversePrompt = reversePrompt
    return self
  }

  public func build() -> T {
    return T.init(
      seed: seed ?? defaults.seed,
      numThreads: numThreads ?? defaults.numThreads,
      numTokens: numTokens ?? defaults.numTokens,
      hyperparameters: hyperparameters.build(),
      reversePrompt: reversePrompt ?? defaults.reversePrompt
    )
  }
}

// MARK: - Params Builders

class SessionConfigParamsBuilder: ObjCxxParamsBuilder {
  let sessionConfig: SessionConfig
  let mode: _LlamaSessionMode

  init(sessionConfig: SessionConfig, mode: _LlamaSessionMode) {
    self.mode = mode
    self.sessionConfig = sessionConfig
  }

  func build(for modelURL: URL) -> _LlamaSessionParams {
    let params = _LlamaSessionParams.defaultParams(withModelPath: modelURL.path, mode: mode)
    params.numberOfThreads = Int32(sessionConfig.numThreads)
    params.numberOfTokens = Int32(sessionConfig.numTokens)

    if let seed = sessionConfig.seed { params.seed = seed }
    params.contextSize = Int32(sessionConfig.hyperparameters.contextSize)
    params.batchSize = Int32(sessionConfig.hyperparameters.batchSize)
    params.lastNTokensToPenalize = Int32(sessionConfig.hyperparameters.lastNTokensToPenalize)
    params.topP = Float(sessionConfig.hyperparameters.topP)
    params.topK = Int32(sessionConfig.hyperparameters.topK)
    params.temp = Float(sessionConfig.hyperparameters.temperature)
    params.repeatPenalty = Float(sessionConfig.hyperparameters.repeatPenalty)

    return params
  }
}

extension SessionConfig {
  static var defaultNumThreads: UInt {
    let processorCount = UInt(ProcessInfo().activeProcessorCount)
    // Account for main thread and worker thread. Specifying all active processors seems to introduce a lot of contention.
    let maxAvailableProcessors = processorCount - 2
    // Experimentally 6 also seems like a pretty good number.
    return min(maxAvailableProcessors, 6)
  }
}

let defaultSessionConfig = {
  let params = _LlamaSessionParams.defaultParams(withModelPath: "", mode: .regular)
  return SessionConfig(
    seed: params.seed == -1 ? nil : params.seed,
    numThreads: UInt(params.numberOfThreads),
    numTokens: UInt(params.numberOfTokens),
    hyperparameters: Hyperparameters(
      contextSize: UInt(params.contextSize),
      batchSize: UInt(params.batchSize),
      lastNTokensToPenalize: UInt(params.lastNTokensToPenalize),
      topK: UInt(params.topK),
      topP: Double(params.topP),
      temperature: Double(params.temp),
      repeatPenalty: Double(params.repeatPenalty)
    ),
    reversePrompt: nil
  )
}()
