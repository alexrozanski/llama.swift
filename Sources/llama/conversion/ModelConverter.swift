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
  struct PythonScriptFile {
    let name: String
    let `extension` = "py"

    var filename: String {
      return "\(name).\(`extension`)"
    }
  }

  enum Script {
    case convertPyTorchToGgml
    case convertGPT4AllToGgml
    case convertUnversionedGgmlToGgml
    case dummy

    var scriptFile: PythonScriptFile {
      switch self {
      case .convertPyTorchToGgml:
        return PythonScriptFile(name:"convert-pth-to-ggml")
      case .convertGPT4AllToGgml:
        return PythonScriptFile(name:"convert-gpt4all-to-ggml")
      case .convertUnversionedGgmlToGgml:
        return PythonScriptFile(name:"convert-unversioned-ggml-to-ggml")
      case .dummy:
        return PythonScriptFile(name:"dummy")
      }
    }

    var url: URL? {
      return Bundle.module.url(forResource: scriptFile.name, withExtension: scriptFile.extension)
    }

    var deps: [String] {
      switch self {
      case .convertPyTorchToGgml:
        return ["numpy", "sentencepiece", "torch"]
      case .convertGPT4AllToGgml:
        return ["sentencepiece"]
      case .convertUnversionedGgmlToGgml:
        return ["sentencepiece"]
      case .dummy:
        return ["numpy", "sentencepiece", "torch"]
      }
    }
  }

  public init() {}

  // MARK: - Validation

  public func validateConversionData(_ data: ConvertPyTorchToGgmlConversionData, requiredFiles: inout [ModelConversionFile]?) -> Result<ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>, ConvertPyTorchToGgmlConversionData.ValidationError> {
    return ConvertPyTorchToGgmlConversion.validate(data, requiredFiles: &requiredFiles)
  }

  // MARK: - Conversion

  public func canRunConversion() async throws -> Bool {
    return try await ModelConversionUtils.checkConversionEnvironment(input: ()).isSuccess
  }

    public func makeConversionPipeline(
      with data: ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>
    ) -> ModelConversionPipeline<
      ConvertPyTorchToGgmlConversionStep,
      ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>,
      ConvertPyTorchToGgmlConversionResult
    > {
      return ConvertPyTorchToGgmlConversion().makeConversionPipeline()
    }

  // MARK: - Quantization

  public func quantizeModel(from sourceFileURL: URL, to destinationFileURL: URL) async throws {
    return try await withCheckedThrowingContinuation { continuation in
      do {
        try _LlamaModelUtils.quantizeModel(withSourceFileURL: sourceFileURL, destFileURL: destinationFileURL, quantizationType: .Q4_0)
      } catch {
        continuation.resume(throwing: error)
      }
    }
  }
}
