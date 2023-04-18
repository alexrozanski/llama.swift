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

protocol ModelConversion<
  DataType,
  ValidatedDataType,
  ConversionStep,
  ValidationError,
  ResultType
> where DataType: ModelConversionData<ValidationError>, ValidationError: Error {
  associatedtype DataType // Input data type
  associatedtype ValidatedDataType // Data type returned from validation (may be the same as DataType)
  associatedtype ConversionStep // The conversion step type. Probably an enum
  associatedtype ConversionPipelineInputType // The input type to the conversion pipeline. This should probably contain ValidatedDataType
  associatedtype ValidationError // The Error type for errors returned from validation
  associatedtype ResultType // The result type from the conversion operation

  // Steps
  static var conversionSteps: [ConversionStep] { get }

  // Validation
  static func validate(
    _ data: DataType,
    returning outRequiredFiles: inout [ModelConversionFile]?
  ) -> Result<ValidatedModelConversionData<ValidatedDataType>, ValidationError>

  // Pipeline
  func makeConversionPipeline() -> ModelConversionPipeline<ConversionStep, ConversionPipelineInputType, ResultType>
}

// Define an additional type that can only be constructed internally by llama.swift
// to ensure that this data has beeen validated by validate(...).
public struct ValidatedModelConversionData<DataType> {
  public let validated: DataType

  internal init(validated: DataType) {
    self.validated = validated
  }
}
