//
//  SessionConfig.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation

public protocol SessionConfig {
  var numTokens: UInt { get }
  var reversePrompt: String? { get }
  var seed: Int32? { get }
}

public class GeneralSessionConfig: SessionConfig {
  public let numThreads: UInt
  public let numTokens: UInt
  public let reversePrompt: String?
  public let seed: Int32?

  public static var `default`: Self {
    let processorCount = UInt(ProcessInfo().activeProcessorCount)
    // Account for main thread and worker thread. Specifying all active processors seems to introduce a lot of contention.
    let maxAvailableProcessors = processorCount - 2
    // Experimentally 6 also seems like a pretty good number.
    let numThreads = min(maxAvailableProcessors, 6)

    return Self.init(
      numThreads: numThreads,
      numTokens: 512, // 512 is default in llama.cpp
      reversePrompt: nil,
      seed: nil
    )
  }

  required public init(
    numThreads: UInt,
    numTokens: UInt,
    reversePrompt: String? = nil,
    seed: Int32? = nil
  ) {
    self.numThreads = numThreads
    self.numTokens = numTokens
    self.reversePrompt = reversePrompt
    self.seed = seed
  }
}
