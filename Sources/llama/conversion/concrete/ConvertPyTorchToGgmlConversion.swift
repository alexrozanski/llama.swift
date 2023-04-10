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
  case setUpEnvironment
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

public struct ConvertPyTorchToGgmlConversionPipelineInput {
  public enum ConversionBehavior {
    case alongsideInputFile
    // Symlinks model to `directory` before converting, then leaves converted file(s) inside this directory.
    case inOtherDirectory(_ directory: URL)
  }

  public let data: ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>
  public let conversionBehavior: ConversionBehavior

  public init(data: ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>, conversionBehavior: ConversionBehavior) {
    self.data = data
    self.conversionBehavior = conversionBehavior
  }
}

fileprivate struct ConvertPyTorchToGgmlConversionConfiguredEnvironment {
  let directoryURL: URL
}

final class ConvertPyTorchToGgmlConversion: ModelConversion {
  // MARK: - Steps

  static var conversionSteps: [ConvertPyTorchToGgmlConversionStep] {
    return ConvertPyTorchToGgmlConversionStep.allCases
  }

  // MARK: - Validation

  static func requiredFiles(for data: ConvertPyTorchToGgmlConversionData) -> [URL] {
    return requiredFilesWithLocations(for: data).map { $0.url }
  }

  private struct FileLocation {
    enum Location {
      case modelDirectory
      case parentDirectory
    }

    let url: URL
    let location: Location
  }

  private static func requiredFilesWithLocations(for data: ConvertPyTorchToGgmlConversionData) -> [FileLocation] {
    let checkpointFiles = (0..<data.modelType.numPyTorchModelParts).map { checkpointFileName(i: $0) }
    let expectedFilesInDirectory = [paramsFileName] + checkpointFiles
    let expectedFilesInParentDirectory = [tokenizerFileName]
    return expectedFilesInDirectory.map {
      FileLocation(url: data.directoryURL.appendingPathComponent($0), location: .modelDirectory)
    } + expectedFilesInParentDirectory.map {
      FileLocation(url: data.directoryURL.deletingLastPathComponent().appendingPathComponent($0), location: .parentDirectory)
    }
  }

  static func validate(
    _ data: ConvertPyTorchToGgmlConversionData,
    requiredFiles: inout [ModelConversionFile]?
  ) -> Result<ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>, ConvertPyTorchToGgmlConversionData.ValidationError> {
    let requiredFileURLs = self.requiredFiles(for: data)

    var missingFilenames: [String] = []
    var requiredFileState = [ModelConversionFile]()

    for fileURL in requiredFileURLs {
      let foundFile = FileManager.default.fileExists(atPath: fileURL.path)
      requiredFileState.append(ModelConversionFile(url: fileURL, found: foundFile))
      if !foundFile {
        missingFilenames.append(fileURL.lastPathComponent)
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
    ConvertPyTorchToGgmlConversionPipelineInput,
    ConvertPyTorchToGgmlConversionResult
  > {
    return ModelConversionPipeline(
      pipeline:
        chainFront(
          makeCheckEnvironmentStep(),
          chainFront(
            makeSetupEnvironmentStep(),
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

  // MARK: - Conversion Steps

  private func makeCheckEnvironmentStep() -> ModelConversionStep<
    ConvertPyTorchToGgmlConversionStep,
    ConvertPyTorchToGgmlConversionPipelineInput,
    ConvertPyTorchToGgmlConversionPipelineInput
  > {
    return ModelConversionStep(type: .checkEnvironment, executionHandler: { input, command, stdout, stderr, cancel in
      return try await ModelConversionUtils.checkConversionEnvironment(
        input: input,
        connectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr, cancel: cancel)
      )
    }, cleanUpHandler: { _ in return true })
  }

  private func makeSetupEnvironmentStep() -> ModelConversionStep<
    ConvertPyTorchToGgmlConversionStep,
    ConvertPyTorchToGgmlConversionPipelineInput,
    ConvertPyTorchToGgmlConversionConfiguredEnvironment
  > {
    return ModelConversionStep(type: .setUpEnvironment, executionHandler: { input, command, stdout, stderr, cancel in
      let validated = input.data.validated
      let directoryURL: URL
      switch input.conversionBehavior {
      case .alongsideInputFile:
        directoryURL = validated.directoryURL
      case .inOtherDirectory(let otherDirectoryURL):
        for fileLocation in ConvertPyTorchToGgmlConversion.requiredFilesWithLocations(for: validated) {
          let filename = fileLocation.url.lastPathComponent
          let destinationDirectory: URL
          switch fileLocation.location {
          case .modelDirectory:
            destinationDirectory = otherDirectoryURL.appendingPathComponent(validated.modelType.subdirectoryName, isDirectory: true)
          case .parentDirectory:
            destinationDirectory = otherDirectoryURL
          }

          let mkdirStatus = try await ModelConversionUtils.run(
            Process.Command("mkdir", arguments: ["-p", destinationDirectory.path]),
            commandConnectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr, cancel: cancel)
          )
          if !mkdirStatus.isSuccess {
            return .failure(exitCode: mkdirStatus.exitCode)
          }

          let destinationFile = destinationDirectory.appendingPathComponent(filename)
          let lnStatus = try await ModelConversionUtils.run(
            Process.Command("ln", arguments: ["-s", fileLocation.url.path, destinationFile.path]),
            commandConnectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr, cancel: cancel)
          )
          if !lnStatus.isSuccess {
            return .failure(exitCode: lnStatus.exitCode)
          }
        }
        directoryURL = otherDirectoryURL.appendingPathComponent(validated.modelType.subdirectoryName, isDirectory: true)
      }

      let environment = ConvertPyTorchToGgmlConversionConfiguredEnvironment(directoryURL: directoryURL)
      return try await ModelConversionUtils.installPythonDependencies(
        input: environment,
        dependencies: PythonScript.convertPyTorchToGgml.deps,
        connectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr, cancel: cancel)
      )
    }, cleanUpHandler: { _ in
      // Shouldn't remove these as they may have been installed anyway.
      return true
    })
  }

  private func makeCheckDependenciesStep() -> ModelConversionStep<
    ConvertPyTorchToGgmlConversionStep,
    ConvertPyTorchToGgmlConversionConfiguredEnvironment,
    ConvertPyTorchToGgmlConversionConfiguredEnvironment
  > {
    return ModelConversionStep(type: .checkDependencies, executionHandler: { input, command, stdout, stderr, cancel in
      return try await ModelConversionUtils.checkInstalledPythonDependencies(
        input: input,
        dependencies: PythonScript.convertPyTorchToGgml.deps,
        connectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr, cancel: cancel)
      )
    }, cleanUpHandler: { _ in return true })
  }

  private func makeConvertFromPyTorchToGgmlStep() -> ModelConversionStep<
    ConvertPyTorchToGgmlConversionStep,
    ConvertPyTorchToGgmlConversionConfiguredEnvironment,
    URL
  > {
    return ModelConversionStep(type: .convertModel, executionHandler: { input, command, stdout, stderr, cancel in
      let inputDirectoryURL = input.directoryURL
      // Hardcode FP16 format for now, like in llama.cpp
      let format = "1"
      let convertStatus = try await ModelConversionUtils.runPythonScript(
        .dummy,
        arguments: [inputDirectoryURL.path, format],
        commandConnectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr, cancel: cancel)
      )
      if !convertStatus.isSuccess {
        return .failure(exitCode: convertStatus.exitCode)
      }

      let resultFilename = "ggml-model-f16.bin"
      let resultFileURL: URL
      if #available(macOS 13.0, iOS 16.0, *) {
        resultFileURL = inputDirectoryURL.appending(path: resultFilename, directoryHint: .isDirectory)
      } else {
        resultFileURL = inputDirectoryURL.appendingPathComponent(resultFilename, isDirectory: true)
      }

      let fileExistsStatus = try await ModelConversionUtils.run(
        Process.Command("test", arguments: ["-f", resultFileURL.path]),
        commandConnectors: CommandConnectors(command: command, stdout: stdout, stderr: stderr, cancel: cancel)
      )
      if !fileExistsStatus.isSuccess {
        return .failure(exitCode: fileExistsStatus.exitCode)
      }

      return .success(result: resultFileURL)
    }, cleanUpHandler: { unQuantizedGgmlFileURL in
      if let unQuantizedGgmlFileURL {
        // Since this is quantized to a new file by the quantize step it is fine to do this
        try FileManager.default.removeItem(at: unQuantizedGgmlFileURL)
      }
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
      executionHandler: { convertedModelURL, command, _, _, _ in
        // TODO: capture stdout and stderr and print
        command("Quantizing model (this may take a few minutes)...")

        let outputBaseURL = URL(fileURLWithPath: (convertedModelURL.path as NSString).deletingLastPathComponent)
        let outputFilename = "ggml-model-q4_0.bin"
        let outputURL: URL
        if #available(macOS 13.0, iOS 16.0, *) {
          outputURL = outputBaseURL.appending(path: outputFilename, directoryHint: .notDirectory)
        } else {
          outputURL = outputBaseURL.appendingPathComponent(outputFilename, isDirectory: false)
        }
        try await ModelConverter().quantizeModel(from: convertedModelURL, to: outputURL)

        return .success(result: ConvertPyTorchToGgmlConversionResult(outputFileURL: outputURL))
      },
      cleanUpHandler: { _ in
        return true
      }
    )
  }
}

fileprivate extension ModelType {
  var subdirectoryName: String {
    switch self {
    case .unknown: return ""
    case .size7B: return "7B"
    case .size13B: return "13B"
    case .size30B: return "30B"
    case .size65B: return "65B"
    }
  }
}
