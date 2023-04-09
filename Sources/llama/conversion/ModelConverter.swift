//
//  ModelConverter.swift
//  llama
//
//  Created by Alex Rozanski on 05/04/2023.
//

import Foundation
import Coquille
import llamaObjCxx

public struct CommandConnectors {
  public typealias CommandConnector = (String) -> Void
  public typealias StdoutConnector = (String) -> Void
  public typealias StderrConnector = (String) -> Void

  public let command: CommandConnector?
  public let stdout: StdoutConnector?
  public let stderr: StderrConnector?

  public init(
    command: CommandConnector?,
    stdout: StdoutConnector?,
    stderr: StderrConnector?
  ) {
    self.command = command
    self.stdout = stdout
    self.stderr = stderr
  }
}

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
    return try await canRunConversion(nil).isSuccess
  }

  public func canRunConversion(_ connectors: CommandConnectors? = nil) async throws -> ModelConversionStatus {
    return try await run("which python3", commandConnectors: connectors)
  }

  public func installDependencies(_ connectors: CommandConnectors? = nil) async throws -> ModelConversionStatus {
    return try await run(Coquille.Process.Command("python3", arguments: ["-u", "-m", "pip", "install"] + Script.convertPyTorchToGgml.deps), commandConnectors: connectors)
  }

  public func checkInstalledDependencies(_ connectors: CommandConnectors? = nil) async throws -> ModelConversionStatus {
    for dep in Script.convertPyTorchToGgml.deps {
      let status = try await run(Coquille.Process.Command("python3", arguments: ["-u", "-m", "pip", "show", dep]), commandConnectors: connectors)
      if !status.isSuccess {
        return status
      }
    }
    return .success
  }

  public func convert(
    with data: ValidatedModelConversionData<ConvertPyTorchToGgmlConversionData>,
    result: inout ConvertPyTorchToGgmlConversionResult?,
    commandConnectors: CommandConnectors? = nil
  ) async throws -> ModelConversionStatus {
    let conversion = ConvertPyTorchToGgmlConversion(data: data)
    defer {
      conversion.cleanUp()
    }
    return try await conversion.run(from: self, result: &result, commandConnectors: commandConnectors)
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

  // MARK: - Internal

  func run(_ command: Coquille.Process.Command, commandConnectors: CommandConnectors? = nil) async throws -> ModelConversionStatus {
    commandConnectors?.command?([[command.name], command.arguments].flatMap { $0 }.joined(separator: " "))
    return try await Process(command: command, stdout: commandConnectors?.stdout, stderr: commandConnectors?.stderr).run().toModelConversionStatus()
  }

  func run(_ commandString: String, commandConnectors: CommandConnectors? = nil) async throws -> ModelConversionStatus {
    commandConnectors?.command?(commandString)
    return try await Process(commandString: commandString, stdout: commandConnectors?.stdout, stderr: commandConnectors?.stderr).run().toModelConversionStatus()
  }
}

fileprivate extension Coquille.Process.Status {
  func toModelConversionStatus() -> ModelConversionStatus {
    switch self {
    case .success:
      return .success
    case .failure(let code):
      return .failure(exitCode: code)
    }
  }
}
