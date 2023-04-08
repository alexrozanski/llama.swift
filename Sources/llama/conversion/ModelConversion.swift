//
//  ModelConversion.swift
//  
//
//  Created by Alex Rozanski on 08/04/2023.
//

import Foundation

public struct ModelConversionFile {
  public let url: URL
  public let found: Bool
}

public enum ModelConversionStatus {
  case success
  case failure(exitCode: Int32)

  public var isSuccess: Bool {
    switch self {
    case .success:
      return true
    case .failure:
      return false
    }
  }

  public var exitCode: Int32 {
    switch self {
    case .success: return 0
    case .failure(exitCode: let exitCode): return exitCode
    }
  }
}

public protocol ModelConversionData<ValidationError> where ValidationError: Error {
  associatedtype ValidationError
}

protocol ModelConversion<DataType, ValidationError> where DataType: ModelConversionData<ValidationError> {
  associatedtype DataType
  associatedtype ValidationError

  static func requiredFiles(for data: DataType) -> [URL]
  static func validate(_ data: DataType, requiredFiles: inout [ModelConversionFile]?) -> Result<ValidatedModelConversionData<DataType>, ValidationError>

  func run(from modelConverter: ModelConverter, commandConnectors: CommandConnectors?) async throws -> ModelConversionStatus

  func cleanUp()
}

public struct ValidatedModelConversionData<DataType> where DataType: ModelConversionData {
  public let data: DataType

  internal init(data: DataType) {
    self.data = data
  }
}
