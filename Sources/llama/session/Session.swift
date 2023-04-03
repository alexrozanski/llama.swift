//
//  Session.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation
import llamaObjCxx

public enum SessionState {
  case notStarted
  case loadingModel
  case readyToPredict
  case predicting
  case error(Error)
}

public struct SessionContext {
  public struct Token {
    private let objCxxToken: _LlamaSessionContextToken

    public var value: Int32 { return objCxxToken.token }
    public var string: String { return objCxxToken.string }

    internal init(objCxxToken: _LlamaSessionContextToken) {
      self.objCxxToken = objCxxToken
    }
  }

  public private(set) lazy var tokens: [Token]? = {
    return objCxxContext.tokens?.map { Token(objCxxToken: $0) }
  }()
  
  public var contextString: String? {
    return objCxxContext.contextString
  }

  private let objCxxContext: _LlamaSessionContext

  internal init(objCxxContext: _LlamaSessionContext) {
    self.objCxxContext = objCxxContext
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
  typealias UpdatedContextHandler = (SessionContext) -> Void

  // MARK: - State

  var state: SessionState { get }
  var stateChangeHandler: StateChangeHandler? { get set }

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

  // Posted when the context changes -- note that this doesn't give an initial value; this can
  // be loaded with currentContext().
  var updatedContextHandler: UpdatedContextHandler? { get set }

  func currentContext() async throws -> SessionContext
}
