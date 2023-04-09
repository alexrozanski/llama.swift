//
//  CommandConnectors.swift
//  
//
//  Created by Alex Rozanski on 09/04/2023.
//

import Foundation

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
