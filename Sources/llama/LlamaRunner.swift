//
//  LlamaRunner.swift
//  llama
//
//  Created by Alex Rozanski on 12/03/2023.
//

import Foundation
import llamaObjCxx

public class LlamaRunner {
  public struct Config {
    public let numThreads: UInt
    public let numTokens: UInt
    public let reversePrompt: String?
    public let seed: Int32?

    public static let `default` = Config(numThreads: 8, numTokens: 512, reversePrompt: nil, seed: nil)

    public init(numThreads: UInt, numTokens: UInt, reversePrompt: String? = nil, seed: Int32? = nil) {
      self.numThreads = numThreads
      self.numTokens = numTokens
      self.reversePrompt = reversePrompt
      self.seed = seed
    }

    fileprivate func toBridgeConfig() -> _LlamaRunnerBridgeConfig {
      let _config = _LlamaRunnerBridgeConfig()
      _config.numberOfThreads = numThreads
      _config.numberOfTokens = numTokens
      _config.reversePrompt = reversePrompt
      _config.seed = seed ?? 0
      return _config
    }
  }

  public enum RunState {
    case notStarted
    case initializing
    case generatingOutput
    case completed
    case failed(error: Error)
  }

  public let modelURL: URL

  private lazy var bridge = _LlamaRunnerBridge(modelPath: modelURL.path)

  public init(modelURL: URL) {
    self.modelURL = modelURL
  }

  // Async-based run() function.
  public func run(
    with prompt: String,
    config: Config = .default,
    stateChangeHandler: ((RunState) -> Void)? = nil
  ) -> AsyncThrowingStream<String, Error> {
    return AsyncThrowingStream<String, Error> { continuation in
      stateChangeHandler?(.notStarted)

      bridge.run(
        withPrompt: prompt,
        config: config.toBridgeConfig(),
        eventHandler: { event in
          event.match(
            startedLoadingModel: {
              stateChangeHandler?(.initializing)
            },
            finishedLoadingModel: {},
            startedGeneratingOutput: {
              stateChangeHandler?(.generatingOutput)
            },
            outputToken: { token in
              continuation.yield(token)
            },
            completed: {
              stateChangeHandler?(.completed)
              continuation.finish()
            },
            failed: { error in
              stateChangeHandler?(.failed(error: error))
              continuation.finish(throwing: error)
            }
          )
        },
        eventHandlerQueue: DispatchQueue.main
      )
    }
  }

  // Closure-based run() function.
  public func run(
    with prompt: String,
    config: Config = .default,
    tokenHandler: @escaping (String) -> Void,
    stateChangeHandler: ((RunState) -> Void)? = nil
  ) {
    stateChangeHandler?(.notStarted)

    bridge.run(
      withPrompt: prompt,
      config: config.toBridgeConfig(),
      eventHandler: { event in
        event.match(
          startedLoadingModel: {
            stateChangeHandler?(.initializing)
          },
          finishedLoadingModel: {},
          startedGeneratingOutput: {
            stateChangeHandler?(.generatingOutput)
          },
          outputToken: { token in
            tokenHandler(token)
          },
          completed: {
            stateChangeHandler?(.completed)
          },
          failed: { error in
            stateChangeHandler?(.failed(error: error))
          }
        )
      },
      eventHandlerQueue: DispatchQueue.main
    )
  }
}
