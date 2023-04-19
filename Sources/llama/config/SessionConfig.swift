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
  public let contextSize: UInt?
  public let batchSize: UInt?
  public let lastNTokensToPenalize: UInt?
  public let topK: UInt?
  // Should be between 0 and 1
  public let topP: Double?
  public let temperature: Double?
  public let repeatPenalty: Double?

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
  public let seed: Int32?

  // Number of threads to run prediction on.
  public let numThreads: UInt?

  // Number of tokens to predict for each run.
  public let numTokens: UInt

  // Model configuration
  public let hyperparameters: Hyperparameters

  public let reversePrompt: String?

  required init(
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
