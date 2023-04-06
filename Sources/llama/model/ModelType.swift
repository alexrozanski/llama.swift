//
//  ModelType.swift
//  llama
//
//  Created by Alex Rozanski on 02/04/2023.
//

import Foundation

public enum ModelType {
  case unknown
  case size7B
  case size13B
  case size30B
  case size65B

  public var numPyTorchModelParts: Int {
    switch self {
    case .unknown:
      return 0
    case .size7B:
      return 1
    case .size13B:
      return 2
    case .size30B:
      return 4
    case .size65B:
      return 8
    }
  }
}
