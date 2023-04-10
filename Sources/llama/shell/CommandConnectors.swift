//
//  CommandConnectors.swift
//  
//
//  Created by Alex Rozanski on 09/04/2023.
//

import Foundation
import Combine

public struct CommandConnectors {
  public typealias CommandConnector = (String) -> Void
  public typealias StdoutConnector = (String) -> Void
  public typealias StderrConnector = (String) -> Void

  public let command: CommandConnector?
  public let stdout: StdoutConnector?
  public let stderr: StderrConnector?
  public let cancel: CurrentValueSubject<Bool, Never>

  public init(
    command: CommandConnector?,
    stdout: StdoutConnector?,
    stderr: StderrConnector?,
    cancel: CurrentValueSubject<Bool, Never>
  ) {
    self.command = command
    self.stdout = stdout
    self.stderr = stderr
    self.cancel = cancel
  }
}

public func makeEmptyConnectors() -> CommandConnectors {
  return CommandConnectors(command: { _ in }, stdout: { _ in }, stderr: { _ in }, cancel: CurrentValueSubject<Bool, Never>(false))
}
