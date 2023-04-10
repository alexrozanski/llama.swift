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

public enum ModelConversionStatus<ResultType> {
  case success(result: ResultType)
  case failure(exitCode: Int32)
  case cancelled

  public var isSuccess: Bool {
    switch self {
    case .success:
      return true
    case .failure, .cancelled:
      return false
    }
  }

  public var exitCode: Int32 {
    switch self {
    case .success: return 0
    case .failure(exitCode: let exitCode): return exitCode
    case .cancelled: return 1
    }
  }
}

public protocol ModelConversionData<ValidationError> where ValidationError: Error {
  associatedtype ValidationError
}

protocol ModelConversion<DataType, ConversionStep, ValidationError, ResultType> where DataType: ModelConversionData<ValidationError> {
  associatedtype DataType
  associatedtype ConversionStep
  associatedtype ConversionPipelineInputType
  associatedtype ValidationError
  associatedtype ResultType

  // Steps
  static var conversionSteps: [ConversionStep] { get }

  // Validation
  static func validate(
    _ data: DataType,
    requiredFiles: inout [ModelConversionFile]?
  ) -> Result<ValidatedModelConversionData<DataType>, ValidationError>

  // Pipeline
  func makeConversionPipeline() -> ModelConversionPipeline<ConversionStep, ConversionPipelineInputType, ResultType>
}

public struct ValidatedModelConversionData<DataType> where DataType: ModelConversionData {
  public let validated: DataType

  internal init(validated: DataType) {
    self.validated = validated
  }
}
