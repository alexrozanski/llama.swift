//
//  Session.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation

public enum SessionState {
  case notStarted
  case loadingModel
  case readyToPredict
  case predicting
  case error(Error)
}

public struct SessionContext {
  public let contextString: String?
  public let tokens: [Int64]?

  internal init(contextString: String?, tokens: [Int64]?) {
    self.contextString = contextString
    self.tokens = tokens
  }
}

public enum PredictionState {
  case notStarted
  case predicting
  case cancelled
  case finished
  case error(Error)
}

public protocol PredictionCancellable {
  func cancel()
}

public protocol Session {
  typealias StateChangeHandler = (SessionState) -> Void
  typealias TokenHandler = (String) -> Void
  typealias PredictionStateChangeHandler = (PredictionState) -> Void

  // MARK: - State

  var state: SessionState { get }
  var stateChangeHandler: StateChangeHandler? { get }

  // MARK: - Prediction

  // Run prediction to generate tokens.
  func predict(with prompt: String) -> AsyncStream<String>

  // Supports state changes.
  func predict(
    with prompt: String,
    stateChangeHandler: @escaping PredictionStateChangeHandler,
    handlerQueue: DispatchQueue?
  ) -> AsyncStream<String>

  // Supports cancellation of prediction.
  func predict(
    with prompt: String,
    tokenHandler: @escaping TokenHandler,
    stateChangeHandler: @escaping PredictionStateChangeHandler,
    handlerQueue: DispatchQueue?
  ) -> PredictionCancellable

  // MARK: - Diagnostics

  func currentContext() async throws -> SessionContext
}
