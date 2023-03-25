//
//  Session.swift
//  llama
//
//  Created by Alex Rozanski on 23/03/2023.
//

import Foundation
import llamaObjCxx

public class Session: NSObject {
  enum Mode {
    case regular
    case instructional

    fileprivate func toObjCxxMode() -> LlamaSessionMode {
      switch self {
      case .regular:
        return .regular
      case .instructional:
        return .instructional
      }
    }
  }

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

    fileprivate func toObjCxxConfig() -> _LlamaSessionConfig {
      let _config = _LlamaSessionConfig()
      _config.numberOfThreads = numThreads
      _config.numberOfTokens = numTokens
      _config.reversePrompt = reversePrompt
      _config.seed = seed ?? 0
      return _config
    }
  }

  public enum State {
    case notStarted
    case loadingModel
    case readyToPredict
    case predicting
    case error(Error)
  }

  public typealias StateChangeHandler = (State) -> Void

  let modelURL: URL
  let mode: Mode
  public let config: Config

  public private(set) var state: State = .notStarted {
    didSet {
      stateChangeHandler?(state)
    }
  }

  public let stateChangeHandler: StateChangeHandler?

  private lazy var _session = _LlamaSession(
    modelPath: modelURL.path,
    mode: mode.toObjCxxMode(),
    config: config.toObjCxxConfig(),
    delegate: self
  )

  init(
    modelURL: URL,
    mode: Mode,
    config: Config,
    stateChangeHandler: StateChangeHandler? = nil
  ) {
    self.modelURL = modelURL
    self.mode = mode
    self.stateChangeHandler = stateChangeHandler
    self.config = config
  }

  public func runPrediction(with prompt: String) -> AsyncThrowingStream<String, Error> {
    return AsyncThrowingStream<String, Error> { continuation in
      _session.runPrediction(
        withPrompt: prompt,
        tokenHandler: { token in
          continuation.yield(token)
        },
        completionHandler: {
          self.state = .readyToPredict
          continuation.finish()
        },
        failureHandler: { error in
          self.state = .error(error)
          continuation.finish(throwing: error)
        }
      )
    }
  }
}

extension Session: _LlamaSessionDelegate {
  public func didStartLoadingModel(in session: _LlamaSession) {
    state = .loadingModel
  }

  public func didLoadModel(in session: _LlamaSession) {
    state = .readyToPredict
  }

  public func didStartPredicting(in session: _LlamaSession) {
    state = .predicting
  }

  public func didFinishPredicting(in session: _LlamaSession) {
    state = .readyToPredict
  }

  public func session(_ session: _LlamaSession, didMoveToErrorStateWithError error: Error) {
    state = .error(error)
  }
}
