//
//  ModelConversionStep.swift
//  
//
//  Created by Alex Rozanski on 09/04/2023.
//

import Foundation
import Combine

public class AnyConversionStep<ConversionStep> {
  @Published public var state: ModelConversionStep<ConversionStep, Void, Any>.State = .notStarted

  private var _type: () -> ConversionStep
  private var _commandOutput: () -> PassthroughSubject<String, Never>
  private var _stdoutOutput: () -> PassthroughSubject<String, Never>
  private var _stderrOutput: () -> PassthroughSubject<String, Never>

  public var type: ConversionStep {
    return _type()
  }

  public var commandOutput: PassthroughSubject<String, Never> {
    return _commandOutput()
  }

  public var stdoutOutput: PassthroughSubject<String, Never> {
    return _stdoutOutput()
  }

  public var stderrOutput: PassthroughSubject<String, Never> {
    return _stderrOutput()
  }

  private var subscriptions = Set<AnyCancellable>()

  init<InputType, ResultType>(wrapped: ModelConversionStep<ConversionStep, InputType, ResultType>) {
    _type = { return wrapped.type }
    _commandOutput = { return wrapped.commandOutput }
    _stdoutOutput = { return wrapped.stdoutOutput }
    _stderrOutput = { return wrapped.stderrOutput }

    wrapped.$state.sink { [weak self] newState in
      switch newState {
      case .notStarted: self?.state = .notStarted
      case .running: self?.state = .running
      case .skipped: self?.state = .skipped
      case .finished(result: let result):
        switch result {
        case .success:
          self?.state = .finished(result: .success(.success(result: ())))
        case .failure(let error):
          self?.state = .finished(result: .failure(error))
        }
      }
    }.store(in: &subscriptions)
  }
}

public class ModelConversionStep<ConversionStep, InputType, ResultType> {
  typealias ExecutionHandler = (
    _ input: InputType,
    _ command: @escaping (String) -> Void,
    _ stdout: @escaping (String) -> Void,
    _ stderr: @escaping (String) -> Void
  ) async throws -> ModelConversionStatus<ResultType>

  public enum OutputType {
    case command
    case stdout
    case stderr

    public var isCommand: Bool {
      switch self {
      case .command:
        return true
      case .stdout, .stderr:
        return false
      }
    }
  }

  public enum State {
    case notStarted
    case skipped
    case running
    case finished(result: Result<ModelConversionStatus<ResultType>, Error>)

    public var canStart: Bool {
      switch self {
      case .notStarted: return true
      case .skipped, .running, .finished: return false
      }
    }

    public var isFinished: Bool {
      switch self {
      case .notStarted, .running: return false
      case .skipped, .finished: return true
      }
    }
  }

  @Published private(set) var state: State = .notStarted

  public let commandOutput = PassthroughSubject<String, Never>()
  public let stdoutOutput = PassthroughSubject<String, Never>()
  public let stderrOutput = PassthroughSubject<String, Never>()

  public let type: ConversionStep
  let executionHandler: ExecutionHandler

  init(type: ConversionStep, executionHandler: @escaping ExecutionHandler) {
    self.type = type
    self.executionHandler = executionHandler
  }

  func execute(with input: InputType) async throws -> Result<ModelConversionStatus<ResultType>, Error> {
    guard state.canStart else { return .failure(NSError()) }

    state = .running

    func makeAppend(prefix: String?, outputType: OutputType) -> ((String) -> Void) {
      return { [weak self] string in
        DispatchQueue.main.async { [weak self] in
          self?.sendOutput(string: string, outputType: outputType)
        }
      }
    }

    let stderr = makeAppend(prefix: nil, outputType: .stderr)
    do {
      let status = try await executionHandler(
        input,
        makeAppend(prefix: "> ", outputType: .command),
        makeAppend(prefix: nil, outputType: .stdout),
        stderr
      )
      let result = Result<ModelConversionStatus<ResultType>, Error>.success(status)
      await MainActor.run {
        // .success() is a bit misleading because the command could have failed, but
        // .success() indicates that *executing* the command succeeded.
        state = .finished(result: result)
      }
      return result
    } catch {
      let result = Result<ModelConversionStatus<ResultType>, Error>.failure(error)
      await MainActor.run {
        stderr(error.localizedDescription)
        if let underlyingError = (error as NSError).userInfo[NSUnderlyingErrorKey] as? Error {
          stderr("\n\n\(underlyingError.localizedDescription)")
        }
        state = .finished(result: .failure(error))
      }
      return result
    }
  }

  func skip() {
    guard state.canStart else { return }

    sendOutput(string: "Skipped step", outputType: .stdout)
    state = .skipped
  }

  private func sendOutput(string: String, outputType: OutputType) {
    let outputString: String
    if outputType.isCommand {
      outputString = "> \(string)"
    } else {
      outputString = string
    }

    switch outputType {
    case .command:
      commandOutput.send(outputString)
    case .stdout:
      stdoutOutput.send(outputString)
    case .stderr:
      stderrOutput.send(outputString)
    }
  }
}
