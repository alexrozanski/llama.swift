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

public enum ConvertPyTorchToGgmlConversionStep: CaseIterable {
  case checkEnvironment
  case installDependencies
  case checkDependencies
  case convertModel
  case quantizeModel
}

public struct ConvertPyTorchToGgmlConversionResult {
  public let outputFileURL: URL

  public func cleanUp() throws {
    try FileManager.default.removeItem(at: outputFileURL)
  }
}

final class ConvertPyTorchToGgmlConversion: ModelConversion {
  static func requiredFiles(for data: ConvertPyTorchToGgmlConversionData) -> [URL] {
    let checkpointFiles = (0..<data.modelType.numPyTorchModelParts).map { checkpointFileName(i: $0) }
    let expectedFiles = [paramsFileName, tokenizerFileName] + checkpointFiles
    return expectedFiles.map { data.directoryURL.appendingPathComponent($0) }
  }

  // MARK: - Steps

  static var conversionSteps: [ConvertPyTorchToGgmlConversionStep] {
    return ConvertPyTorchToGgmlConversionStep.allCases
  }

  // MARK: - Validation

  static func validate(
    _ data: ConvertPyTorchToGgmlConversionData,
    requiredFiles: inout [ModelConversionFile]?
  ) -> Result<ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>, ConvertPyTorchToGgmlConversionData.ValidationError> {
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
      return .success(ValidatedModelConversionData(validated: data))
    }
  }

  // MARK: - Conversion

  func makeConversionPipeline() -> ModelConversionPipeline<
    ConvertPyTorchToGgmlConversionStep,
    ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>,
    ConvertPyTorchToGgmlConversionResult
  > {
    return ModelConversionPipeline(
      pipeline:
        chainFront(
          makeCheckEnvironmentStep(),
          chainFront(
            makeInstallDependenciesStep(),
            chainFront(
              makeCheckDependenciesStep(),
              chainFront(
                makeConvertFromPyTorchToGgmlStep(),
                UnconnectedConversionStep(
                  step: makeQuantizeStep()
                )
              )
            )
          )
        )
    )
  }

  private func makeCheckEnvironmentStep() -> ModelConversionStep<
    ConvertPyTorchToGgmlConversionStep,
    ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>,
    ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>
  > {
    return ModelConversionStep(type: .checkEnvironment, executionHandler: { input, command, stdout, stderr in
      return try await ModelConversionUtils.checkConversionEnvironment(
        input: input,
        connectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr)
      )
    }, cleanUpHandler: { _ in return true })
  }

  private func makeInstallDependenciesStep() -> ModelConversionStep<
    ConvertPyTorchToGgmlConversionStep,
    ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>,
    ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>
  > {
    return ModelConversionStep(type: .installDependencies, executionHandler: { input, command, stdout, stderr in
      return try await ModelConversionUtils.installPythonDependencies(
        input: input,
        dependencies: ModelConverter.Script.convertPyTorchToGgml.deps,
        connectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr)
      )
    }, cleanUpHandler: { _ in
      // Shouldn't remove these as they may have been installed anyway.
      return true
    })
  }

  private func makeCheckDependenciesStep() -> ModelConversionStep<
    ConvertPyTorchToGgmlConversionStep,
    ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>,
    ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>
  > {
    return ModelConversionStep(type: .checkDependencies, executionHandler: { input, command, stdout, stderr in
      return try await ModelConversionUtils.checkInstalledPythonDependencies(
        input: input,
        dependencies: ModelConverter.Script.convertPyTorchToGgml.deps,
        connectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr)
      )
    }, cleanUpHandler: { _ in return true })
  }

  private func makeConvertFromPyTorchToGgmlStep() -> ModelConversionStep<
    ConvertPyTorchToGgmlConversionStep,
    ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>,
    URL
  > {
    return ModelConversionStep(type: .convertModel, executionHandler: { input, command, stdout, stderr in
      let script = ModelConverter.Script.dummy
      guard let url = script.url else { return .failure(exitCode: 1) }

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

      let inputDirectoryURL = input.validated.directoryURL
      let convertStatus = try await ModelConversionUtils.run(
        Coquille.Process.Command(
          "python3",
          arguments: [
            "-u",
            scriptFileURL.path,
            input.validated.directoryURL.path
          ]
        ),
        commandConnectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr)
      )
      if !convertStatus.isSuccess {
        return .failure(exitCode: convertStatus.exitCode)
      }

      let resultFilename = "ggml-model-1.bin"
      let resultFileURL: URL
      if #available(macOS 13.0, iOS 16.0, *) {
        resultFileURL = inputDirectoryURL.appending(path: resultFilename, directoryHint: .isDirectory)
      } else {
        resultFileURL = inputDirectoryURL.appendingPathComponent(resultFilename, isDirectory: true)
      }

      let fileExistsStatus = try await ModelConversionUtils.run(
        Process.Command("test", arguments: ["-f", resultFileURL.path]),
        commandConnectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr)
      )
      if !fileExistsStatus.isSuccess {
        return .failure(exitCode: convertStatus.exitCode)
      }

      return .success(result: resultFileURL)
    }, cleanUpHandler: { unQuantizedGgmlFileURL in
      // Since this is quantized to a new file by the quantize step it is fine to do this
      try FileManager.default.removeItem(at: unQuantizedGgmlFileURL)
      return true
    })
  }

  private func makeQuantizeStep() -> ModelConversionStep<
    ConvertPyTorchToGgmlConversionStep,
    URL,
    ConvertPyTorchToGgmlConversionResult
  > {
    return ModelConversionStep(
      type: .quantizeModel,
      executionHandler: { convertedModelURL, _, _, _ in
        let fileURL = URL(fileURLWithPath: (convertedModelURL.path as NSString).deletingLastPathComponent).appendingPathComponent("ggml-model-q4_0-dummy.bin")
        try String(" ").write(to: fileURL, atomically: true, encoding: .utf8)
        return .success(result: ConvertPyTorchToGgmlConversionResult(outputFileURL: fileURL))
      },
      cleanUpHandler: { _ in
        return true
      }
    )
  }
}
