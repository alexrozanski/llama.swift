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

  public static func validate(_ data: Data, requiredFiles: inout [ModelConversionFile]?) -> Result<Void, Data.ValidationError> {
    let paramsFile = data.directoryURL.appendingPathComponent(paramsFileName)
    let tokenizerFile = data.directoryURL.appendingPathComponent(tokenizerFileName)

    var result: Result<Void, Data.ValidationError>? = nil
    var requiredFileState = [ModelConversionFile]()

    let foundParams = FileManager.default.fileExists(atPath: paramsFile.path)
    requiredFileState.append(ModelConversionFile(url: paramsFile, found: foundParams))
    if !foundParams && result == nil {
      result = .failure(.missingParamsFile(filename: paramsFileName))
    }

    let foundTokenizerFile = FileManager.default.fileExists(atPath: tokenizerFile.path)
    requiredFileState.append(ModelConversionFile(url: tokenizerFile, found: foundParams))
    if !foundTokenizerFile && result == nil {
      result = .failure(.missingTokenizerFile(filename: tokenizerFileName))
    }

    for i in (0..<data.modelType.numPyTorchModelParts) {
      let checkpointFileName = checkpointFileName(i: i)
      let checkpointFile = data.directoryURL.appendingPathComponent(checkpointFileName)
      let foundCheckpointFile = FileManager.default.fileExists(atPath: checkpointFile.path)
      requiredFileState.append(ModelConversionFile(url: checkpointFile, found: foundCheckpointFile))
      if !foundCheckpointFile && result == nil {
        result = .failure(.missingPyTorchCheckpoint(filename: checkpointFileName))
      }
    }

    requiredFiles = requiredFileState

    return .success(())
  }
}
