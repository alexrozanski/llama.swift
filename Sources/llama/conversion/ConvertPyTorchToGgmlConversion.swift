//
//  ConvertPyTorchToGgmlConversion.swift
//  
//
//  Created by Alex Rozanski on 06/04/2023.
//

import Foundation
import Coquille

private let paramsFileName = "params.json"
private let tokenizerFileName = "tokenizer.model"

private func checkpointFileName(i: Int) -> String {
  return "consolidated.0\(i).pth"
}

public struct ConvertPyTorchToGgmlConversionData: ModelConversionData {
  public enum ValidationError: Error {
    case missingFiles(filenames: [String])
  }

  public let modelType: ModelType
  public let directoryURL: URL

  public init(modelType: ModelType, directoryURL: URL) {
    self.modelType = modelType
    self.directoryURL = directoryURL
  }
}

final class ConvertPyTorchToGgmlConversion: ModelConversion {
  let data: ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>
  init(data: ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>) {
    self.data = data
  }

  static func requiredFiles(for data: ConvertPyTorchToGgmlConversionData) -> [URL] {
    let checkpointFiles = (0..<data.modelType.numPyTorchModelParts).map { checkpointFileName(i: $0) }
    let expectedFiles = [paramsFileName, tokenizerFileName] + checkpointFiles
    return expectedFiles.map { data.directoryURL.appendingPathComponent($0) }
  }

  static func validate(_ data: ConvertPyTorchToGgmlConversionData, requiredFiles: inout [ModelConversionFile]?) -> Result<ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>, ConvertPyTorchToGgmlConversionData.ValidationError> {
    let paramsFile = data.directoryURL.appendingPathComponent(paramsFileName)
    let tokenizerFile = data.directoryURL.appendingPathComponent(tokenizerFileName)

    var missingFilenames: [String] = []
    var requiredFileState = [ModelConversionFile]()

    let foundParams = FileManager.default.fileExists(atPath: paramsFile.path)
    requiredFileState.append(ModelConversionFile(url: paramsFile, found: foundParams))
    if !foundParams {
      missingFilenames.append(paramsFileName)
    }

    let foundTokenizerFile = FileManager.default.fileExists(atPath: tokenizerFile.path)
    requiredFileState.append(ModelConversionFile(url: tokenizerFile, found: foundParams))
    if !foundTokenizerFile {
      missingFilenames.append(tokenizerFileName)
    }

    for i in (0..<data.modelType.numPyTorchModelParts) {
      let checkpointFileName = checkpointFileName(i: i)
      let checkpointFile = data.directoryURL.appendingPathComponent(checkpointFileName)
      let foundCheckpointFile = FileManager.default.fileExists(atPath: checkpointFile.path)
      requiredFileState.append(ModelConversionFile(url: checkpointFile, found: foundCheckpointFile))
      if !foundCheckpointFile {
        missingFilenames.append(checkpointFileName)
      }
    }

    requiredFiles = requiredFileState

    if !missingFilenames.isEmpty {
      return .failure(.missingFiles(filenames: missingFilenames))
    } else {
      return .success(ValidatedModelConversionData(data: data))
    }
  }

  func run(from modelConverter: ModelConverter, commandConnectors: CommandConnectors? = nil) async throws -> ModelConversionStatus {
    let script = ModelConverter.Script.convertPyTorchToGgml
    guard let url = script.url else { return .failure(exitCode: -1) }

    let temporaryDirectoryURL: URL
    if #available(macOS 13.0, iOS 16.0, *) {
      temporaryDirectoryURL = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString, directoryHint: .isDirectory)
    } else {
      temporaryDirectoryURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
    }
    try FileManager.default.createDirectory(at: temporaryDirectoryURL, withIntermediateDirectories: true)

    let scriptFileURL: URL
    if #available(macOS 13.0, iOS 16.0, *) {
      scriptFileURL = temporaryDirectoryURL.appending(path: script.scriptFile.filename, directoryHint: .notDirectory)
    } else {
      scriptFileURL = temporaryDirectoryURL.appendingPathComponent(script.scriptFile.filename, isDirectory: false)
    }

    let contents = try String(contentsOf: url)
    try contents.write(to: scriptFileURL, atomically: true, encoding: .utf8)

    return try await modelConverter.run(Coquille.Process.Command("python3", arguments: ["-u", scriptFileURL.path]), commandConnectors: commandConnectors)
  }
}
