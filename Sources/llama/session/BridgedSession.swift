//
//  LlamaSession.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

protocol ObjCxxParamsBuilder {
  func build() -> _LlamaSessionParams
}

class BridgedPredictionCancellable: PredictionCancellable {
  let objCxxHandle: _LlamaSessionPredictionHandle

  init(objCxxHandle: _LlamaSessionPredictionHandle) {
    self.objCxxHandle = objCxxHandle
  }

  func cancel() {
    objCxxHandle.cancel()
  }
}

class BridgedSession: NSObject, Session, _LlamaSessionDelegate {
  let paramsBuilder: ObjCxxParamsBuilder

  private(set) var state: SessionState = .notStarted {
    didSet {
      stateChangeHandler?(state)
    }
  }

  let stateChangeHandler: StateChangeHandler?

  private lazy var _session = _LlamaSession(
    params: paramsBuilder.build(),
    delegate: self
  )

  init(
    paramsBuilder: ObjCxxParamsBuilder,
    stateChangeHandler: StateChangeHandler?
  ) {
    self.paramsBuilder = paramsBuilder
    self.stateChangeHandler = stateChangeHandler
  }

  func predict(with prompt: String) -> AsyncStream<String> {
    return AsyncStream<String> { continuation in
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
        },
        handlerQueue: .main
      )
    }
  }

  @MainActor func predict(with prompt: String, receiveToken: @escaping (String) -> Void, on receiveQueue: DispatchQueue?) -> PredictionCancellable {
    let objCxxHandle = _session.runPrediction(
      withPrompt: prompt,
      tokenHandler: { token in
        (receiveQueue ?? .main).async {
          receiveToken(token)
        }
      },
      completionHandler: {
        self.state = .readyToPredict
      },
      failureHandler: { error in
        self.state = .error(error)
      },
      handlerQueue: .main
    )

    return BridgedPredictionCancellable(objCxxHandle: objCxxHandle)
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
