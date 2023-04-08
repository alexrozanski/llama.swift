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

  // MARK: - Validation

  public static func validateData<Data>(_ data: Data, requiredFiles: inout [ModelConversionFile]?) -> Result<Void, Data.ValidationError> where Data: ModelConversionData {
    return Data.ModelConversionType.validate(data, requiredFiles: &requiredFiles)
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

  public static func convert() {
    run(script: .dummy)
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
  func toModelConverterStatus() -> ModelConverter.Status {
    switch self {
    case .success:
      return .success
    case .failure(let code):
      return .failure(exitCode: code)
    }
  }
}
