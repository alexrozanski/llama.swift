//
//  ModelConversionPipeline.swift
//  
//
//  Created by Alex Rozanski on 09/04/2023.
//

import Foundation
import Combine

public class ModelConversionPipeline<StepType, InputType, ResultType> {
  public enum State {
    case notRunning
    case running
    case failed
    case finished(result: ResultType)

    public var isRunning: Bool {
      switch self {
      case .notRunning, .failed, .finished: return false
      case .running: return true
      }
    }
  }

  public var steps: [AnyConversionStep<StepType>] {
    return pipeline.steps
  }

  @Published private(set) var state: State = .notRunning
  let pipeline: any ChainedConversionStep<StepType, InputType, ResultType>

  init(pipeline: any ChainedConversionStep<StepType, InputType, ResultType>) {
    self.pipeline = pipeline
  }

  public var canStart: Bool {
    switch state {
    case .notRunning: return true
    case .running, .failed, .finished: return false
    }
  }

  public func run(with input: InputType) async throws {
    guard canStart else { return }

    await MainActor.run {
      state = .running
    }

    let result = try await pipeline.execute(with: input)
    await MainActor.run {
      switch result {
      case .success(let executionResult):
        switch executionResult {
        case .success(result: let result):
          state = .finished(result: result)
        case .failure:
          state = .failed
        }
      case .failure:
        state = .failed
      }
    }
  }

  public func stop() {}
}
