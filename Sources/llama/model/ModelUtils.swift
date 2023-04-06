//
//  ModelUtils.swift
//  llama
//
//  Created by Alex Rozanski on 05/04/2023.
//

import Foundation
import llamaObjCxx

public class ModelUtils {
  private init() {}

  public static func getModelType(forFileAt fileURL: URL) throws -> ModelType {
    var modelType: _LlamaModelType = .typeUnknown
    try _LlamaModelUtils.loadModelTypeForFile(at: fileURL, outModelType: &modelType)
    switch modelType {
    case .typeUnknown:
      return .unknown
    case .type7B:
      return .size7B
    case .type13B:
      return .size13B
    case .type30B:
      return .size30B
    case .type65B:
      return .size65B
    default:
      return .unknown
    }
  }

  public static func validateModel(fileURL: URL) throws {
    do {
      _ = try getModelType(forFileAt: fileURL)
    } catch {
      let error = error as NSError

      // Since we get the model type by loading the file, failures should be `failedToLoadModel`.
      guard error.domain == _LlamaErrorDomain, error.code == _LlamaErrorCode.failedToLoadModel.rawValue else {
        throw error
      }

      // Retag this as `failedToValidateModel` for a more consistent API
      throw NSError(domain: _LlamaErrorDomain, code: _LlamaErrorCode.failedToValidateModel.rawValue, userInfo: error.userInfo)
    }
  }
}
