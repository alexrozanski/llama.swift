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
  let session = inference.createSession(
    with: Session.Config(numThreads: 8, numTokens: 15, seed: 1920476),
    stateChangeHandler: { state in
      print("state change:", state)
    }
  )

  let tokenStream = session.runPrediction(with: "Building a website can be done in 10 simple steps:")

  do {
    for try await token in tokenStream {
      print(token, terminator: "")
    }
  } catch let error {
    print("")
    print("Failed to generate output:", error.localizedDescription)
  }

//  while true {
//    print("Enter prompt: ")
//    guard let prompt = readLine()?.trimmingCharacters(in: .whitespacesAndNewlines), !prompt.isEmpty else {
//      break
//    }
//
//    let tokenStream = LlamaRunner(modelURL: url).run(
//      with: prompt,
//      stateChangeHandler: { state in
//        switch state {
//        case .notStarted:
//          break
//        case .initializing:
//          print("Initializing model... ", terminator: "")
//        case .generatingOutput:
//          print("Done.")
//          print("")
//          print("Generating output...")
//          print("\"", terminator: "")
//        case .completed:
//          print("\"")
//          print("")
//        case .failed:
//          // Handle this in the catch {}
//          break
//        }
//      })
//
//    do {
//      for try await token in tokenStream {
//        print(token, terminator: "")
//      }
//    } catch let error {
//      print("")
//      print("Failed to generate output:", error.localizedDescription)
//    }
//  }
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
