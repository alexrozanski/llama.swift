//
//  ModelConverter.swift
//  llama
//
//  Created by Alex Rozanski on 05/04/2023.
//

import Foundation
import Coquille

public enum ModelConversionOperation {
  case convertPyTorchToGgml
}

public struct ModelConversionFile {
  public let url: URL
  public let found: Bool
}

public protocol ModelConversionData<ModelConversionType, ValidationError> where ModelConversionType: ModelConversion<Self, ValidationError>, ValidationError: Error {
  associatedtype ValidationError
  associatedtype ModelConversionType
}

public protocol ModelConversion<DataType, ValidationError> where DataType: ModelConversionData<Self, ValidationError> {
  associatedtype DataType
  associatedtype ValidationError

  static func requiredFiles(for data: DataType) -> [URL]
  static func validate(_ data: DataType, requiredFiles: inout [ModelConversionFile]?) -> Result<ValidatedModelConversionData<DataType>, ValidationError>
}

public struct ValidatedModelConversionData<DataType> where DataType: ModelConversionData {
  public let data: DataType

  internal init(data: DataType) {
    self.data = data
  }
}

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
  public enum Status {
    case success
    case failure(exitCode: Int32)

    public var isSuccess: Bool {
      switch self {
      case .success:
        return true
      case .failure:
        return false
      }
    }

    public var exitCode: Int32 {
      switch self {
      case .success: return 0
      case .failure(exitCode: let exitCode): return exitCode
      }
    }
  }

  private struct PythonScriptFile {
    let name: String
    let `extension` = "py"

    var filename: String {
      return "\(name).\(`extension`)"
    }
  }

  private enum Script {
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

  // MARK: - Validation

  public static func validateData<DataType>(_ data: DataType, requiredFiles: inout [ModelConversionFile]?) -> Result<ValidatedModelConversionData<DataType>, DataType.ValidationError> where DataType: ModelConversionData {
    return DataType.ModelConversionType.validate(data, requiredFiles: &requiredFiles)
  }

  // MARK: - Conversion

  public static func canRunConversion() async throws -> Bool {
    return try await canRunConversion(nil).isSuccess
  }

  public static func canRunConversion(_ connectors: CommandConnectors? = nil) async throws -> Status {
    return try await run("which python3", commandConnectors: connectors)
  }

  public static func installDependencies(_ connectors: CommandConnectors? = nil) async throws -> Status {
    return try await run(Coquille.Process.Command("python3", arguments: ["-u", "-m", "pip", "install"] + Script.convertPyTorchToGgml.deps), commandConnectors: connectors)
  }

  public static func checkInstalledDependencies(_ connectors: CommandConnectors? = nil) async throws -> Status {
    for dep in Script.convertPyTorchToGgml.deps {
      let status = try await run(Coquille.Process.Command("python3", arguments: ["-u", "-m", "pip", "show", dep]), commandConnectors: connectors)
      if !status.isSuccess {
        return status
      }
    }
    return .success
  }

  public static func convertPyTorchModels(
    with data: ValidatedModelConversionData<ConvertPyTorchToGgmlConversion.Data>,
    commandConnectors: CommandConnectors? = nil
  ) async throws -> Status {
    try await run(script: .dummy, commandConnectors: commandConnectors)
  }

  // MARK: - Private

  private static func run(_ command: Coquille.Process.Command, commandConnectors: CommandConnectors? = nil) async throws -> Status {
    commandConnectors?.command?([[command.name], command.arguments].flatMap { $0 }.joined(separator: " "))
    return try await Process(command: command, stdout: commandConnectors?.stdout, stderr: commandConnectors?.stderr).run().toModelConverterStatus()
  }

  private static func run(_ commandString: String, commandConnectors: CommandConnectors? = nil) async throws -> Status {
    commandConnectors?.command?(commandString)
    return try await Process(commandString: commandString, stdout: commandConnectors?.stdout, stderr: commandConnectors?.stderr).run().toModelConverterStatus()
  }

  private static func run(script: Script, commandConnectors: CommandConnectors? = nil) async throws -> Status {
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

    return try await run(Coquille.Process.Command("python3", arguments: ["-u", scriptFileURL.path]), commandConnectors: commandConnectors)
  }
}

private extension Coquille.Process.Status {
  func toModelConverterStatus() -> ModelConverter.Status {
    switch self {
    case .success:
      return .success
    case .failure(let code):
      return .failure(exitCode: code)
    }
  }
}
