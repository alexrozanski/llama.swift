//
//  ChainedConversionStep.swift
//  
//
//  Created by Alex Rozanski on 09/04/2023.
//

import Foundation

protocol ChainedConversionStep<StepType, InputType, OutputType> {
  associatedtype StepType
  associatedtype InputType
  associatedtype OutputType

  func execute(with input: InputType) async throws -> Result<ModelConversionStatus<OutputType>, Error>
}

func chainFront<StepType, NewInputType, InputType, OutputType>(
  _ front: ModelConversionStep<StepType, NewInputType, InputType>,
  _ step: any ChainedConversionStep<StepType, InputType, OutputType>
) -> any ChainedConversionStep<StepType, NewInputType, OutputType> {
  return ConnectedConversionStep<StepType, NewInputType, InputType, OutputType>(input: front, output: step)
}

class UnconnectedConversionStep<StepType, InputType, OutputType>: ChainedConversionStep {
  let step: ModelConversionStep<StepType, InputType, OutputType>
  init(step: ModelConversionStep<StepType, InputType, OutputType>) {
    self.step = step
  }

  func execute(with input: InputType) async throws -> Result<ModelConversionStatus<OutputType>, Error> {
    return try await step.execute(with: input)
  }
}

class ConnectedConversionStep<StepType, InputType, IO, OutputType>: ChainedConversionStep {
  let input: ModelConversionStep<StepType, InputType, IO>
  let output: any ChainedConversionStep<StepType, IO, OutputType>

  init(input: ModelConversionStep<StepType, InputType, IO>, output: any ChainedConversionStep<StepType, IO, OutputType>) {
    self.input = input
    self.output = output
  }

  func execute(with input: InputType) async throws -> Result<ModelConversionStatus<OutputType>, Error> {
    let status = try await self.input.execute(with: input)
    switch status {
    case .success(let status):
      switch status {
      case .success(result: let result):
        return try await output.execute(with: result)
      case .failure(exitCode: let exitCode):
        return .success(.failure(exitCode: exitCode))
      }
    case .failure(let error):
      return .failure(error)
    }
  }
}
