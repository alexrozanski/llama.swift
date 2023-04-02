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

  // Synchronize state on the main queue.
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

  // MARK: - Model Metrics

  public static func getModelType(forFileAt fileURL: URL) throws -> ModelType {
    var modelType: _LlamaModelType = .typeUnknown
    try _LlamaSession.loadModelTypeForFile(at: fileURL, outModelType: &modelType)
    switch modelType {
    case .typeUnknown:
      return .unknown
    case .type7B:
      return .size7B
    case .type12B:
      return .size12B
    case .type30B:
      return .size30B
    case .type65B:
      return .size65B
    default:
      return .unknown
    }
  }

  // MARK: - Prediction

  func predict(with prompt: String) -> AsyncStream<String> {
    return AsyncStream<String> { continuation in
      _session.runPrediction(
        withPrompt: prompt,
        startHandler: {},
        tokenHandler: { token in
          continuation.yield(token)
        },
        completionHandler: {
          self.state = .readyToPredict
          continuation.finish()
        },
        cancelHandler: {
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

  func predict(
    with prompt: String,
    stateChangeHandler: @escaping PredictionStateChangeHandler,
    handlerQueue: DispatchQueue?
  ) -> AsyncStream<String> {
    let outerHandlerQueue = handlerQueue ?? .main
    return AsyncStream<String> { continuation in
      outerHandlerQueue.async {
        stateChangeHandler(.notStarted)
      }
      _session.runPrediction(
        withPrompt: prompt,
        startHandler: {
          outerHandlerQueue.async {
            stateChangeHandler(.predicting)
          }
        },
        tokenHandler: { token in
          continuation.yield(token)
        },
        completionHandler: {
          self.state = .readyToPredict
          outerHandlerQueue.async {
            stateChangeHandler(.finished)
          }
          continuation.finish()
        },
        cancelHandler: {
          self.state = .readyToPredict
          outerHandlerQueue.async {
            stateChangeHandler(.cancelled)
          }
          continuation.finish()
        },
        failureHandler: { error in
          self.state = .error(error)
          outerHandlerQueue.async {
            stateChangeHandler(.error(error))
          }
        },
        handlerQueue: .main
      )
    }
  }

  func predict(
    with prompt: String,
    tokenHandler: @escaping TokenHandler,
    stateChangeHandler: @escaping PredictionStateChangeHandler,
    handlerQueue: DispatchQueue?
  ) -> PredictionCancellable {
    let outerHandlerQueue = handlerQueue ?? .main
    outerHandlerQueue.async {
      stateChangeHandler(.notStarted)
    }
    let objCxxHandle = _session.runPrediction(
      withPrompt: prompt,
      startHandler: {
        outerHandlerQueue.async {
          stateChangeHandler(.predicting)
        }
      },
      tokenHandler: { token in
        outerHandlerQueue.async {
          tokenHandler(token)
        }
      },
      completionHandler: {
        self.state = .readyToPredict
        outerHandlerQueue.async {
          stateChangeHandler(.finished)
        }
      },
      cancelHandler: {
        self.state = .readyToPredict
        outerHandlerQueue.async {
          stateChangeHandler(.cancelled)
        }
      },
      failureHandler: { error in
        self.state = .error(error)
        outerHandlerQueue.async {
          stateChangeHandler(.error(error))
        }
      },
      handlerQueue: .main
    )

    return BridgedPredictionCancellable(objCxxHandle: objCxxHandle)
  }

  // MARK: - Diagnostics

  func currentContext() async throws -> SessionContext {
    return try await withCheckedThrowingContinuation { continuation in
      _session.getCurrentContext(handler: { objCxxSessionContext in
        let tokens = (objCxxSessionContext.tokens ?? []).map { $0.int64Value }
        continuation.resume(returning: SessionContext(contextString: objCxxSessionContext.contextString, tokens: tokens))
      }, errorHandler: { error in
        continuation.resume(throwing: error)
      })
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
