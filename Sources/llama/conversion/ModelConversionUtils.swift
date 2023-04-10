//
//  ModelConversionUtils.swift
//  
//
//  Created by Alex Rozanski on 09/04/2023.
//

import Foundation
import Coquille

struct PythonScriptFile {
  let name: String
  let `extension` = "py"

  var filename: String {
    return "\(name).\(`extension`)"
  }
}

enum PythonScript {
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

  static func runPythonScript(
    _ script: PythonScript,
    arguments: [String],
    commandConnectors: CommandConnectors? = nil
  ) async throws -> ModelConversionStatus<Void> {
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

    return try await ModelConversionUtils.run(
      Coquille.Process.Command(
        "python3",
        arguments: [
          "-u",
          scriptFileURL.path
        ] + arguments
      ),
      commandConnectors: commandConnectors
    )
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