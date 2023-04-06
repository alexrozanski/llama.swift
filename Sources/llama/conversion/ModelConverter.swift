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

  static func validate(_ data: DataType) -> Bool
}

public protocol ModelConversionData {}

public class ConvertPyTorchToGgmlConversion: ModelConversion {
  public struct Data: ModelConversionData {
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

  public static func validate(_ data: Data) -> Bool {
    let paramsFile = data.directoryURL.appendingPathComponent("params.json")
    let tokenizerFile = data.directoryURL.appendingPathComponent("tokenizer.model")

    guard
      FileManager.default.fileExists(atPath: paramsFile.path),
      FileManager.default.fileExists(atPath: tokenizerFile.path)
    else {
      return false
    }

    for i in (0..<data.modelType.numPyTorchModelParts) {
      if !FileManager.default.fileExists(atPath: data.directoryURL.appendingPathComponent("consolidated.0\(i).pth").path) {
        return false
      }
    }

    return true
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

  public static func validateData(_ data: ModelConversionData) -> Bool {
    if let data = data as? ConvertPyTorchToGgmlConversion.Data {
      return ConvertPyTorchToGgmlConversion.validate(data)
    }

    return false
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
