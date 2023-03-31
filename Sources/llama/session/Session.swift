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

public protocol PredictionCancellable {
  func cancel()
}

public protocol Session {
  typealias StateChangeHandler = (SessionState) -> Void

  var state: SessionState { get }
  var stateChangeHandler: StateChangeHandler? { get }

  // Run prediction to generate tokens.
  func predict(with prompt: String) -> AsyncStream<String>

  // Supports cancellation of prediction.
  func predict(
    with prompt: String,
    receiveToken: @escaping (String) -> Void,
    on receiveQueue: DispatchQueue?
  ) -> PredictionCancellable
}
