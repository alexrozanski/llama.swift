//
//  ModelConverter.swift
//  llama
//
//  Created by Alex Rozanski on 05/04/2023.
//

import Foundation
import Coquille

public enum ModelConversionType {
  case convertPyTorchToGgml
}

public protocol ModelConversion<DataType> where DataType: ModelConversionData {
  associatedtype DataType

  static func validate(_ data: DataType) -> Result<Void, DataType.ValidationError>
}

public protocol ModelConversionData<ModelConversionType, ValidationError> where ModelConversionType: ModelConversion<Self>, ValidationError: Error {
  associatedtype ValidationError
  associatedtype ModelConversionType
}

public class ConvertPyTorchToGgmlConversion: ModelConversion {
  public struct Data: ModelConversionData {
    public typealias ModelConversionType = ConvertPyTorchToGgmlConversion

    public enum ValidationError: Error {
      case missingParamsFile(filename: String)
      case missingTokenizerFile(filename: String)
      case missingPyTorchCheckpoint(filename: String)
    }

    public let modelType: ModelType
    public let directoryURL: URL

    public init(modelType: ModelType, directoryURL: URL) {
      self.modelType = modelType
      self.directoryURL = directoryURL
    }
  }

  let data: Data
  init(data: Data) {
    self.data = data
  }

  public static func validate(_ data: Data) -> Result<Void, Data.ValidationError> {
    let paramsFileName = "params.json"
    let tokenizerFileName = "tokenizer.model"

    let paramsFile = data.directoryURL.appendingPathComponent(paramsFileName)
    let tokenizerFile = data.directoryURL.appendingPathComponent(tokenizerFileName)

    if !FileManager.default.fileExists(atPath: paramsFile.path) {
      return .failure(.missingParamsFile(filename: paramsFileName))
    }
    if !FileManager.default.fileExists(atPath: tokenizerFile.path) {
      return .failure(.missingParamsFile(filename: tokenizerFileName))
    }

    for i in (0..<data.modelType.numPyTorchModelParts) {
      let checkpointFileName = "consolidated.0\(i).pth"
      if !FileManager.default.fileExists(atPath: data.directoryURL.appendingPathComponent(checkpointFileName).path) {
        return .failure(.missingPyTorchCheckpoint(filename: checkpointFileName))
      }
    }
    return .success(())
  }
}

public class ModelConverter {
  private enum Script {
    case convertPyTorchToGgml
    case convertGPT4AllToGgml
    case convertUnversionedGgmlToGgml
    case dummy

    var url: URL? {
      switch self {
      case .convertPyTorchToGgml:
        return Bundle.module.url(forResource: "convert-pth-to-ggml", withExtension: "py")
      case .convertGPT4AllToGgml:
        return Bundle.module.url(forResource: "convert-gpt4all-to-ggml", withExtension: "py")
      case .convertUnversionedGgmlToGgml:
        return Bundle.module.url(forResource: "convert-unversioned-ggml-to-ggml", withExtension: "py")
      case .dummy:
        return Bundle.module.url(forResource: "dummy", withExtension: "py")
      }
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

  public static func canRunConversion() async -> Bool {
    do {
      return try await Process(commandString: "which python3", printStdout: false, printStderr: false).run().isSuccess
    } catch {
      return false
    }
  }

  public static func validateData<Data>(_ data: Data) -> Result<Void, Data.ValidationError> where Data: ModelConversionData {
    return Data.ModelConversionType.validate(data)
  }

  public static func convert() {
    run(script: .dummy)
  }

  private static func run(script: Script) {
    guard let url = script.url else { return }

    do {
      let contents = try String(contentsOf: url)
    } catch {
      print(error)
    }
  }
}
