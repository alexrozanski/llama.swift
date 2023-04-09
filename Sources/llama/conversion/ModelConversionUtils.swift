//
//  ModelConversionUtils.swift
//  
//
//  Created by Alex Rozanski on 09/04/2023.
//

import Foundation
import Coquille

class ModelConversionUtils {
  private init() {}

  static func checkConversionEnvironment<Input>(
    input: Input,
    connectors: CommandConnectors? = nil
  ) async throws -> ModelConversionStatus<Input> {
    let status = try await ModelConversionUtils.run("which python3", commandConnectors: connectors)
    switch status {
    case .success:
      return .success(result: input)
    case .failure(exitCode: let exitCode):
      return .failure(exitCode: exitCode)
    }
  }

  static func installPythonDependencies<Input>(
    input: Input,
    dependencies: [String],
    connectors: CommandConnectors? = nil
  ) async throws -> ModelConversionStatus<Input> {
    let status = try await run(Coquille.Process.Command("python3", arguments: ["-u", "-m", "pip", "install"] + dependencies), commandConnectors: connectors)
    switch status {
    case .success:
      return .success(result: input)
    case .failure(exitCode: let exitCode):
      return .failure(exitCode: exitCode)
    }
  }

  static func checkInstalledPythonDependencies<Input>(
    input: Input,
    dependencies: [String],
    connectors: CommandConnectors? = nil
  ) async throws -> ModelConversionStatus<Input> {
    for dependency in dependencies {
      let status = try await run(Coquille.Process.Command("python3", arguments: ["-u", "-m", "pip", "show", dependency]), commandConnectors: connectors)
      switch status {
      case .success:
        break
      case .failure(exitCode: let exitCode):
        return .failure(exitCode: exitCode)
      }
    }
    return .success(result: input)
  }

  static func run(_ command: Coquille.Process.Command, commandConnectors: CommandConnectors? = nil) async throws -> ModelConversionStatus<Void> {
    commandConnectors?.command?([[command.name], command.arguments].flatMap { $0 }.joined(separator: " "))
    return try await Process(command: command, stdout: commandConnectors?.stdout, stderr: commandConnectors?.stderr).run().toModelConversionStatus()
  }

  static func run(_ commandString: String, commandConnectors: CommandConnectors? = nil) async throws -> ModelConversionStatus<Void> {
    commandConnectors?.command?(commandString)
    return try await Process(commandString: commandString, stdout: commandConnectors?.stdout, stderr: commandConnectors?.stderr).run().toModelConversionStatus()
  }
}

fileprivate extension Coquille.Process.Status {
  func toModelConversionStatus() -> ModelConversionStatus<Void> {
    switch self {
    case .success:
      return .success(result: ())
    case .failure(let code):
      return .failure(exitCode: code)
    }
  }
}
