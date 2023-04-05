// swift-tools-version:5.5

import PackageDescription

let package = Package(
  name: "llama.swift",
  platforms: [
    .macOS(.v10_15),
    .iOS(.v13),
  ],
  products: [
    .library(name: "llama", targets: ["llama"]),
  ],
  targets: [
    .target(
      name: "llama",
      dependencies: ["llamaObjCxx"],
      path: "Sources/llama",
      resources: [
        .copy("resources/convert-gpt4all-to-ggml.py"),
        .copy("resources/convert-pth-to-ggml.py"),
        .copy("resources/convert-unversioned-ggml-to-ggml.py")
      ]),
    .target(
      name: "llamaObjCxx",
      dependencies: [],
      path: "Sources/llamaObjCxx",
      publicHeadersPath: "headers",
      cxxSettings: [
        .headerSearchPath("cpp"),
        .headerSearchPath("session/operations"),
        .headerSearchPath("internal")
      ])
  ],
  cLanguageStandard: .gnu11,
  cxxLanguageStandard: .gnucxx20
)
