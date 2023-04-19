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
  public fileprivate(set) var contextSize: UInt?
  public fileprivate(set) var batchSize: UInt?
  public fileprivate(set) var lastNTokensToPenalize: UInt?
  public fileprivate(set) var topK: UInt?
  // Should be between 0 and 1
  public fileprivate(set) var topP: Double?
  public fileprivate(set) var temperature: Double?
  public fileprivate(set) var repeatPenalty: Double?

  public init(
    contextSize: UInt? = nil,
    batchSize: UInt? = nil,
    lastNTokensToPenalize: UInt? = nil,
    topK: UInt? = nil,
    topP: Double? = nil,
    temperature: Double? = nil,
    repeatPenalty: Double? = nil
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
  public private(set) var numThreads: UInt?

  // Number of tokens to predict for each run.
  public private(set) var numTokens: UInt

  // Model configuration
  public private(set) var hyperparameters: Hyperparameters

  public let reversePrompt: String?

  required public init(
    seed: Int32? = nil,
    numThreads: UInt? = nil,
    numTokens: UInt,
    hyperparameters: Hyperparameters,
    reversePrompt: String? = nil
  ) {
    self.seed = seed
    self.numThreads = numThreads
    self.numTokens = numTokens
    self.hyperparameters = hyperparameters
    self.reversePrompt = reversePrompt
  }

  public func withNumThreads(_ numThreads: UInt?) -> Self {
    self.numThreads = numThreads
    return self
  }

  public func withNumTokens(_ numTokens: UInt) -> Self {
    self.numTokens = numTokens
    return self
  }

  public func withContextSize(_ contextSize: UInt?) -> Self {
    self.hyperparameters.contextSize = contextSize
    return self
  }

  public func withBatchSize(_ batchSize: UInt?) -> Self {
    self.hyperparameters.batchSize = batchSize
    return self
  }

  public func withLastNTokensToPenalize(_ lastNTokensToPenalize: UInt?) -> Self {
    self.hyperparameters.lastNTokensToPenalize = lastNTokensToPenalize
    return self
  }

  public func withTopK(_ topK: UInt?) -> Self {
    self.hyperparameters.topK = topK
    return self
  }

  public func withTopP(_ topP: Double?) -> Self {
    self.hyperparameters.topP = topP
    return self
  }

  public func withTemperature(_ temperature: Double?) -> Self {
    self.hyperparameters.temperature = temperature
    return self
  }

  public func withRepeatPenalty(_ repeatPenalty: Double?) -> Self {
    self.hyperparameters.repeatPenalty = repeatPenalty
    return self
  }
}

class SessionConfigBuilder: ObjCxxParamsBuilder {
  let sessionConfig: SessionConfig
  let mode: _LlamaSessionMode

  init(sessionConfig: SessionConfig, mode: _LlamaSessionMode) {
    self.mode = mode
    self.sessionConfig = sessionConfig
  }

  func build(for modelURL: URL) -> _LlamaSessionParams {
    let params = _LlamaSessionParams.defaultParams(withModelPath: modelURL.path, mode: mode)
    if let numThreads = sessionConfig.numThreads {
      params.numberOfThreads = Int32(numThreads)
    }

    params.numberOfTokens = Int32(sessionConfig.numTokens)

    if let seed = sessionConfig.seed { params.seed = seed }
    if let contextSize = sessionConfig.hyperparameters.contextSize { params.contextSize = Int32(contextSize) }
    if let batchSize = sessionConfig.hyperparameters.batchSize { params.batchSize = Int32(batchSize) }
    if let lastNTokensToPenalize = sessionConfig.hyperparameters.lastNTokensToPenalize { params.lastNTokensToPenalize = Int32(lastNTokensToPenalize) }
    if let topP = sessionConfig.hyperparameters.topP { params.topP = Float(topP) }
    if let topK = sessionConfig.hyperparameters.topK { params.topK = Int32(topK) }
    if let temperature = sessionConfig.hyperparameters.temperature { params.temp = Float(temperature) }
    if let repeatPenalty = sessionConfig.hyperparameters.repeatPenalty { params.repeatPenalty = Float(repeatPenalty) }

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
