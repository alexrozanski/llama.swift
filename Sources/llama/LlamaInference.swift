//
//  LlamaInference.swift
//  llama
//
//  Created by Alex Rozanski on 24/03/2023.
//

import Foundation

public class LlamaInference {
  public let modelURL: URL

  public init(modelURL: URL) {
    self.modelURL = modelURL
  }

  public func createSession(with config: Session.Config, stateChangeHandler: Session.StateChangeHandler?) -> Session {
    return Session(modelURL: modelURL, mode: .regular, config: config, stateChangeHandler: stateChangeHandler)
  }

  public func createInstructionalSession(with config: Session.Config, stateChangeHandler: Session.StateChangeHandler?) -> Session {
    return Session(modelURL: modelURL, mode: .instructional, config: config, stateChangeHandler: stateChangeHandler)
  }
}
