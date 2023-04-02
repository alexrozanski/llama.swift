//
//  Error.swift
//  llama
//
//  Created by Alex Rozanski on 02/04/2023.
//

import Foundation
import llamaObjCxx

public struct LlamaError {
  private init() {}

  // Match to values in _LlamaErrorCode
  public enum Code: Int {
    case failedToLoadModel = -1000
    case failedToPredict = -1001
    case failedToLoadSessionContext = -1002
  }

  public static let domain = _LlamaErrorDomain
}
