//
//  Session.swift
//  llama
//
//  Created by Alex Rozanski on 23/03/2023.
//

import Foundation

public class Session {
  public enum Mode {
    case regular
    case instructional
  }

  public struct Config {
    public let numTokens: UInt
    public let seed: Int32?

    public init(numTokens: UInt, seed: Int32?) {
      self.numTokens = numTokens
      self.seed = seed
    }
  }

  public let mode: Mode
  public let config: Config

  public init(mode: Mode, config: Config) {
    self.mode = mode
    self.config = config
  }
}
