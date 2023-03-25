# llama.swift

[![CI](https://github.com/alexrozanski/llama.swift/actions/workflows/build.yml/badge.svg)](https://github.com/alexrozanski/llama.swift/actions/workflows/build.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A fork of [@ggerganov](https://github.com/ggerganov)'s [llama.cpp](https://github.com/ggerganov/llama.cpp) to use [Facebook's LLaMA](https://github.com/facebookresearch/llama) models in Swift.

See the [main repository](https://github.com/ggerganov/llama.cpp/) for info about the original goals of the project and implementation.

## LLaMA Setup

Clone the repo:

```bash
git clone https://github.com/alexrozanski/llama.swift.git
cd llama.swift
```

Grab the LLaMA model weights and place them in `./models`. `ls` should print something like:

```bash
ls ./models
65B 30B 13B 7B tokenizer_checklist.chk tokenizer.model
```

To convert the LLaMA-7B model and quantize:

```bash
# install Python dependencies
python3 -m pip install torch numpy sentencepiece

# the command-line tools are in `./tools` instead of the repo root like in llama.cpp
cd tools

# convert the 7B model to ggml FP16 format
python3 convert-pth-to-ggml.py ../models/7B/ 1

# quantize the model to 4-bits
make
python3 quantize.py 7B
```

When running the larger models, make sure you have enough disk space to store all of the intermediate files.

## Installation

### Swift Package Manager

Add `llama.swift` to your project using Xcode (File > Add Packages...) or by adding it to your project's `Package.swift` file:

```swift
dependencies: [
  .package(url: "https://github.com/alexrozanski/llama.swift.git", .upToNextMajor(from: "1.0.0"))
]
```

## Usage

### Swift library

To generate output from a prompt, first instantiate a `LlamaRunner` instance with the URL to your LLaMA model file:

```swift
import llama

let url = ... // URL to the ggml-model-q4_0.bin model file
let runner = LlamaRunner(modelURL: url)
```

Generating output is as simple as calling `run()` with your prompt on the `LlamaRunner` instance. Since tokens are generated asynchronously this returns an `AsyncThrowingStream` which you can enumerate over to process tokens as they are returned:

```swift
do {
  for try await token in runner.run(with: "Building a website can be done in 10 simple steps:") {
    print(token, terminator: "")
  }
} catch let error {
  // Handle error
}
```

Note that tokens don't necessarily correspond to a single word, and also include any whitespace and newlines.

#### Configuration

`LlamaRunner.run()` takes an optional `LlamaRunner.Config` instance which lets you control the number of threads inference is run on (default: `8`), the maximum number of tokens returned (default: `512`) and an optional reverse/negative prompt:

```swift
let prompt = "..."
let config = LlamaRunner.Config(numThreads: 8, numTokens: 20, reversePrompt: "...")
let tokenStream = runner.run(with: prompt, config: config)

do {
  for try await token in tokenStream {
    ...
  }
} catch let error {
  ...
}
```

#### State Changes

`LlamaRunner.run()` also takes an optional `stateChangeHandler` closure, which is invoked whenever the run state changes:

```
let prompt = "..."
let tokenStream = runner.run(
  with: prompt,
  config: .init(numThreads: 8, numTokens: 20),
  stateChangeHandler: { state in
    switch state {
      case .notStarted:
        // Initial state
        break
      case .initializing:
        // Loading the model and initializing
        break
      case .generatingOutput:
        // Generating tokens
        break
      case .completed:
        // Completed successfully
        break
      case .failed:
        // Failed. This is also the error thrown by the `AsyncThrowingSequence` returned from `LlamaRunner.run()`
        break
    }
  })
```

#### Closure-based API

If you don't want to use Swift concurrency there is an alternative version of `run()` which returns tokens via a `tokenHandler` closure instead:

```swift
let prompt = "..."
runner.run(
  with: prompt,
  config: ...,
  tokenHandler: { token in
    ...
  },
  stateChangeHandler: ...
)
```

#### Other notes

- Build for Release if you want token generation to be snappy, since `llama` will generate tokens slowly in Debug builds.
- Because of the way the Swift package is structured (and some gaps in my knowledge around exported symbols from modules), including `llama.swift` also leaks the name of the internal module containing the Objective-C/C++ implementation, `llamaObjCxx`, as well as some internal classes prefixed with `_Llama`. Pull requests welcome if you have any ideas on fixing this!


### `llamaTest` app

The repo contains a barebones command-line tool, `llamaTest`, which uses the `llama` Framework to run a simple input loop to run inference on a given input prompt.

- Ensure to set `MODEL_PATH` in `LlamaTest.xcconfig` to point to your `path/to/ggml-model-q4_0.bin` (without quotes or spaces after `MODEL_PATH=`), for example:

```
MODEL_PATH=/path/to/ggml-model-q4_0.bin
```

## Misc

- License: MIT
- Other matters: See the [llama.cpp repo](https://github.com/ggerganov/llama.cpp/).
