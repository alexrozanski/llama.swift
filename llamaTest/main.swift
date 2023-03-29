//
//  main.swift
//  llamaTest
//
//  Created by Alex Rozanski on 12/03/2023.
//

import Foundation
import llama

guard let pathString = Bundle.main.object(forInfoDictionaryKey: "LlamaModelPath") as? String else {
  print("Model path not specified - define in MODEL_PATH")
  exit(1)
}

guard let url = URL(string: pathString), FileManager.default.fileExists(atPath: url.path) else {
  print("Invalid model path, make sure this is a file URL")
  exit(1)
}

// Run Llama

@Sendable func run() async {
  let inference = LlamaInference(modelURL: url)
  let session = inference.createInstructionalSession(
    with: Session.Config(numThreads: 8, numTokens: 512, seed: 1920476),
    stateChangeHandler: { state in
      print("state change:", state)
    }
  )

  while true {
    print("Enter prompt: ")
    guard let prompt = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines), !prompt.isEmpty else {
      break
    }

    let tokenStream = session.runPrediction(with: prompt)
    do {
      for try await token in tokenStream {
        print(token, terminator: "")
      }
    } catch let error {
      print("")
      print("Failed to generate output:", error.localizedDescription)
    }
  }
}

// Run program.
let semaphore = DispatchSemaphore(value: 0)

Task.init {
  await run()
}

// Don't block the main thread to ensure that state changes are still called
// on the main thread.
while semaphore.wait(timeout: .now()) == .timedOut {
  RunLoop.current.run(mode: .default, before: Date(timeIntervalSinceNow: 0))
}
