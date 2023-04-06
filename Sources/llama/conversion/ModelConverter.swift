//
//  ModelConverter.swift
//  llama
//
//  Created by Alex Rozanski on 05/04/2023.
//

import Foundation
import Coquille

public class ModelConverter {
  private enum Script {
    case convertPthToGgml
    case convertGPT4AllToGgml
    case convertUnversionedGgmlToGgml
    case dummy

    var url: URL? {
      switch self {
      case .convertPthToGgml:
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
      case .convertPthToGgml:
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

  public static func canRunConversion() async -> Bool {
    do {
      return try await Process(commandString: "which python3", printStdout: false, printStderr: false).run().isSuccess
    } catch {
      return false
    }
  }

  public static func convert() {
    run(script: .dummy)
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
