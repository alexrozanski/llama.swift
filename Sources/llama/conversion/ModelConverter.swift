//
//  ModelConverter.swift
//  llama
//
//  Created by Alex Rozanski on 05/04/2023.
//

import Foundation
import Coquille
import llamaObjCxx

public class ModelConverter {
  public init() {}

  // MARK: - Validation

  public func validateConversionData(_ data: ConvertPyTorchToGgmlConversionData, returning outRequiredFiles: inout [ModelConversionFile]?) -> Result<ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>, ConvertPyTorchToGgmlConversionData.ValidationError> {
    return ConvertPyTorchToGgmlConversion.validate(data, returning: &outRequiredFiles)
  }

  // MARK: - Conversion

  public func canRunConversion() async throws -> Bool {
    return try await ModelConversionUtils.checkConversionEnvironment(input: (), connectors: makeEmptyConnectors()).isSuccess
  }

    public func makeConversionPipeline() -> ModelConversionPipeline<
      ConvertPyTorchToGgmlConversionStep,
      ConvertPyTorchToGgmlConversionPipelineInput,
      ConvertPyTorchToGgmlConversionResult
    > {
      return ConvertPyTorchToGgmlConversion().makeConversionPipeline()
    }

  // MARK: - Quantization

  public func quantizeModel(from sourceFileURL: URL, to destinationFileURL: URL) async throws {
    return try await withCheckedThrowingContinuation { continuation in
      do {
        try _LlamaModelUtils.quantizeModel(withSourceFileURL: sourceFileURL, destFileURL: destinationFileURL, fileType: .mostlyQ4_0)
        continuation.resume(returning: ())
      } catch {
        continuation.resume(throwing: error)
      }
    }
  }
}
