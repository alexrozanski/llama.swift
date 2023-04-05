//
//  ModelConverter.swift
//  llama
//
//  Created by Alex Rozanski on 05/04/2023.
//

import Foundation

public class ModelConverter {
  public static func convert() {
    guard let url = Bundle.module.url(forResource: "convert-pth-to-ggml", withExtension: "py") else { return }

    do {
      let contents = try String(contentsOf: url)
      print(contents)
    } catch {
      print(error)
    }
  }
}
