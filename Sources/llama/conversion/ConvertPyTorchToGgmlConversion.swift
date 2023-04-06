//
//  ConvertPyTorchToGgmlConversion.swift
//  
//
//  Created by Alex Rozanski on 06/04/2023.
//

import Foundation

private let paramsFileName = "params.json"
private let tokenizerFileName = "tokenizer.model"

private func checkpointFileName(i: Int) -> String {
  return "consolidated.0\(i).pth"
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

  public static func requiredFiles(for data: Data) -> [URL] {
    let checkpointFiles = (0..<data.modelType.numPyTorchModelParts).map { checkpointFileName(i: $0) }
    let expectedFiles = [paramsFileName, tokenizerFileName] + checkpointFiles
    return expectedFiles.map { data.directoryURL.appendingPathComponent($0) }
  }

  public static func validate(_ data: Data) -> Result<Void, Data.ValidationError> {
    let paramsFile = data.directoryURL.appendingPathComponent(paramsFileName)
    let tokenizerFile = data.directoryURL.appendingPathComponent(tokenizerFileName)

    if !FileManager.default.fileExists(atPath: paramsFile.path) {
      return .failure(.missingParamsFile(filename: paramsFileName))
    }
    if !FileManager.default.fileExists(atPath: tokenizerFile.path) {
      return .failure(.missingParamsFile(filename: tokenizerFileName))
    }

    for i in (0..<data.modelType.numPyTorchModelParts) {
      let checkpointFileName = checkpointFileName(i: i)
      if !FileManager.default.fileExists(atPath: data.directoryURL.appendingPathComponent(checkpointFileName).path) {
        return .failure(.missingPyTorchCheckpoint(filename: checkpointFileName))
      }
    }
    return .success(())
  }
}
