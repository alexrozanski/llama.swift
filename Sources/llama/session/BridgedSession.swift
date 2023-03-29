//
//  LlamaSession.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

protocol ObjCxxConfigBuilder {
  func build() -> _LlamaSessionConfig
}

class BridgedSession: NSObject, Session, _LlamaSessionDelegate {
  let modelURL: URL
  let configBuilder: ObjCxxConfigBuilder

  private(set) var state: SessionState = .notStarted

  let stateChangeHandler: StateChangeHandler?

  private lazy var _session = _LlamaSession(
    modelPath: modelURL.path,
    config: configBuilder.build(),
    delegate: self
  )

  init(
    modelURL: URL,
    configBuilder: ObjCxxConfigBuilder,
    stateChangeHandler: StateChangeHandler?
  ) {
    self.modelURL = modelURL
    self.configBuilder = configBuilder
    self.stateChangeHandler = stateChangeHandler
  }

  func predict(with prompt: String) -> AsyncThrowingStream<String, Error> {
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

  // MARK: - _LlamaSessionDelegate

  func didStartLoadingModel(in session: _LlamaSession) {
    state = .loadingModel
  }

  func didLoadModel(in session: _LlamaSession) {
    state = .readyToPredict
  }

  func didStartPredicting(in session: _LlamaSession) {
    state = .predicting
  }

  func didFinishPredicting(in session: _LlamaSession) {
    state = .readyToPredict
  }

  func session(_ session: _LlamaSession, didMoveToErrorStateWithError error: Error) {
    state = .error(error)
  }
}
