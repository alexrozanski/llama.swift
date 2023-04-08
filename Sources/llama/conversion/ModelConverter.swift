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

public struct ModelConversionFile {
  public let url: URL
  public let found: Bool
}

public protocol ModelConversion<DataType> where DataType: ModelConversionData {
  associatedtype DataType

  static func requiredFiles(for data: DataType) -> [URL]
  static func validate(_ data: DataType, requiredFiles: inout [ModelConversionFile]?) -> Result<Void, DataType.ValidationError>
}

public protocol ModelConversionData<ModelConversionType, ValidationError> where ModelConversionType: ModelConversion<Self>, ValidationError: Error {
  associatedtype ValidationError
  associatedtype ModelConversionType
}

public struct CommandConnectors {
  public typealias CommandConnector = (String) -> Void
  public typealias StdoutConnector = (String) -> Void
  public typealias StderrConnector = (String) -> Void
  public typealias ExitCode = (Int32) -> Void

  public let command: CommandConnector?
  public let stdout: StdoutConnector?
  public let stderr: StderrConnector?
  public let exitCode: ExitCode?

  public init(
    command: CommandConnector?,
    stdout: StdoutConnector?,
    stderr: StderrConnector?,
    exitCode: ExitCode?
  ) {
    self.command = command
    self.stdout = stdout
    self.stderr = stderr
    self.exitCode = exitCode
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

  public static func canRunConversion(_ connectors: CommandConnectors? = nil) async -> Bool {
    do {
      return try await run("which python3", commandConnectors: connectors).isSuccess
    } catch {
      return false
    }
  }

  public static func validateData<Data>(_ data: Data, requiredFiles: inout [ModelConversionFile]?) -> Result<Void, Data.ValidationError> where Data: ModelConversionData {
    return Data.ModelConversionType.validate(data, requiredFiles: &requiredFiles)
  }

  public static func convert() {
    run(script: .dummy)
  }

  private static func run(_ command: Coquille.Process.Command, commandConnectors: CommandConnectors? = nil) async throws -> Coquille.Process.Status {
    commandConnectors?.command?([[command.name], command.arguments].flatMap { $0 }.joined(separator: " "))
    let status = try await Process(commandString: "which python3", printStdout: false, printStderr: false).run()
    commandConnectors?.exitCode?(status.toExitCode())
    return status
  }

  private static func run(_ commandString: String, commandConnectors: CommandConnectors? = nil) async throws -> Coquille.Process.Status {
    commandConnectors?.command?(commandString)
    let status = try await Process(commandString: commandString, printStdout: false, printStderr: false).run()
    commandConnectors?.exitCode?(status.toExitCode())
    return status
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

private extension Coquille.Process.Status {
  func toExitCode() -> Int32 {
    switch self {
    case .success:
      return 0
    case .failure(let code):
      return code
    }
  }
}
