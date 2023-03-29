//
//  SessionConfig.swift
//  llama
//
//  Created by Alex Rozanski on 29/03/2023.
//

import Foundation

public protocol SessionConfig {
  var numTokens: UInt { get }
  var reversePrompt: String? { get }
  var seed: Int32? { get }
}

public class GeneralSessionConfig: SessionConfig {
  public let numTokens: UInt
  public let reversePrompt: String?
  public let seed: Int32?

  public init(numTokens: UInt, reversePrompt: String? = nil, seed: Int32? = nil) {
    self.numTokens = numTokens
    self.reversePrompt = reversePrompt
    self.seed = seed
  }
}
