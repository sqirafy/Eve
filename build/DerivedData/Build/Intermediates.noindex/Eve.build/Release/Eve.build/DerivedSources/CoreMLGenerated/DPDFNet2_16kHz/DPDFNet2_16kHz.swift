//
// DPDFNet2_16kHz.swift
//
// This file was automatically generated and should not be edited.
//

import CoreML


/// Model Prediction Input Type
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
class DPDFNet2_16kHzInput : MLFeatureProvider {

    /// spec as 1 × 1 × 161 × 2 4-dimensional array of floats
    var spec: MLMultiArray

    var featureNames: Set<String> { ["spec"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "spec" {
            return MLFeatureValue(multiArray: spec)
        }
        return nil
    }

    init(spec: MLMultiArray) {
        self.spec = spec
    }

    convenience init(spec: MLShapedArray<Float>) {
        self.init(spec: MLMultiArray(spec))
    }

}


/// Model Prediction Output Type
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
class DPDFNet2_16kHzOutput : MLFeatureProvider {

    /// Source provided by CoreML
    private let provider : MLFeatureProvider

    /// spec_e as 1 × 1 × 161 × 2 4-dimensional array of floats
    var spec_e: MLMultiArray {
        provider.featureValue(for: "spec_e")!.multiArrayValue!
    }

    /// spec_e as 1 × 1 × 161 × 2 4-dimensional array of floats
    var spec_eShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(spec_e)
    }

    var featureNames: Set<String> {
        provider.featureNames
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        provider.featureValue(for: featureName)
    }

    init(spec_e: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["spec_e" : MLFeatureValue(multiArray: spec_e)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}

/// Model Prediction State Type
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
final class DPDFNet2_16kHzState {
    enum Name: String, CaseIterable {
        case gru_state = "gru_state"
    }

    let handle: MLState

    init(handle: MLState) {
        self.handle = handle
    }

    func withMultiArray<R>(for stateName: Name, _ body: (MLMultiArray) throws -> R) rethrows -> R {
        try handle.withMultiArray(for: stateName.rawValue, body)
    }
}

/// Class for model loading and prediction
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
class DPDFNet2_16kHz {
    let model: MLModel

    /// URL of model assuming it was installed in the same bundle as this class
    class var urlOfModelInThisBundle : URL {
        let bundle = Bundle(for: self)
        return bundle.url(forResource: "DPDFNet2_16kHz", withExtension:"mlmodelc")!
    }

    /**
        Construct DPDFNet2_16kHz instance with an existing MLModel object.

        Usually the application does not use this initializer unless it makes a subclass of DPDFNet2_16kHz.
        Such application may want to use `MLModel(contentsOfURL:configuration:)` and `DPDFNet2_16kHz.urlOfModelInThisBundle` to create a MLModel object to pass-in.

        - parameters:
          - model: MLModel object
    */
    init(model: MLModel) {
        self.model = model
    }

    /**
        Construct a model with configuration

        - parameters:
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    convenience init(configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        try self.init(contentsOf: type(of:self).urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct DPDFNet2_16kHz instance with explicit path to mlmodelc file
        - parameters:
           - modelURL: the file url of the model

        - throws: an NSError object that describes the problem
    */
    convenience init(contentsOf modelURL: URL) throws {
        try self.init(model: MLModel(contentsOf: modelURL))
    }

    /**
        Construct a model with URL of the .mlmodelc directory and configuration

        - parameters:
           - modelURL: the file url of the model
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    convenience init(contentsOf modelURL: URL, configuration: MLModelConfiguration) throws {
        try self.init(model: MLModel(contentsOf: modelURL, configuration: configuration))
    }

    /**
        Construct DPDFNet2_16kHz instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<DPDFNet2_16kHz, Error>) -> Void) {
        load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration, completionHandler: handler)
    }

    /**
        Construct DPDFNet2_16kHz instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
    */
    class func load(configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> DPDFNet2_16kHz {
        try await load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct DPDFNet2_16kHz instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<DPDFNet2_16kHz, Error>) -> Void) {
        MLModel.load(contentsOf: modelURL, configuration: configuration) { result in
            switch result {
            case .failure(let error):
                handler(.failure(error))
            case .success(let model):
                handler(.success(DPDFNet2_16kHz(model: model)))
            }
        }
    }

    /**
        Construct DPDFNet2_16kHz instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
    */
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> DPDFNet2_16kHz {
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        return DPDFNet2_16kHz(model: model)
    }

    /**
        Make a new state.

        Core ML framework will allocate the state buffers declared in the model.

        The allocated state buffers are initialized to zeros. To initialize with different values, use `.withMultiArray(for:)` to get the mutable `MLMultiArray`-view to the state buffer.

        ```swift
        let state = model.makeState()
        state.withMultiArray(for: .gru_state) { stateMultiArray in
            stateMultiArray[0] = 0.42
        }
        ```
    */
    func makeState() -> DPDFNet2_16kHzState {
        DPDFNet2_16kHzState(handle: model.makeState())
    }

    /**
        Make a prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as DPDFNet2_16kHzInput
           - state: the state that the prediction will use and update.

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as DPDFNet2_16kHzOutput
    */
    func prediction(input: DPDFNet2_16kHzInput, using state: DPDFNet2_16kHzState) throws -> DPDFNet2_16kHzOutput {
        try prediction(input: input, using: state, options: MLPredictionOptions())
    }

    /**
        Make a prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as DPDFNet2_16kHzInput
           - state: the state that the prediction will use and update.
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as DPDFNet2_16kHzOutput
    */
    func prediction(input: DPDFNet2_16kHzInput, using state: DPDFNet2_16kHzState, options: MLPredictionOptions) throws -> DPDFNet2_16kHzOutput {
        let outFeatures = try model.prediction(from: input, using: state.handle, options: options)
        return DPDFNet2_16kHzOutput(features: outFeatures)
    }

    /**
        Make an asynchronous prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as DPDFNet2_16kHzInput
           - state: the state that the prediction will use and update.
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as DPDFNet2_16kHzOutput
    */
    func prediction(input: DPDFNet2_16kHzInput, using state: DPDFNet2_16kHzState, options: MLPredictionOptions = MLPredictionOptions()) async throws -> DPDFNet2_16kHzOutput {
        let outFeatures = try await model.prediction(from: input, using: state.handle, options: options)
        return DPDFNet2_16kHzOutput(features: outFeatures)
    }

    /**
        Make a prediction using the convenience interface

        It uses the default function if the model has multiple functions.

        - parameters:
            - spec: 1 × 1 × 161 × 2 4-dimensional array of floats
            - state: the state that the prediction will use and update.

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as DPDFNet2_16kHzOutput
    */
    func prediction(spec: MLMultiArray, using state: DPDFNet2_16kHzState) throws -> DPDFNet2_16kHzOutput {
        let input_ = DPDFNet2_16kHzInput(spec: spec)
        return try prediction(input: input_, using: state)
    }

    /**
        Make a prediction using the convenience interface

        It uses the default function if the model has multiple functions.

        - parameters:
            - spec: 1 × 1 × 161 × 2 4-dimensional array of floats
            - state: the state that the prediction will use and update.

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as DPDFNet2_16kHzOutput
    */

    func prediction(spec: MLShapedArray<Float>, using state: DPDFNet2_16kHzState) throws -> DPDFNet2_16kHzOutput {
        let input_ = DPDFNet2_16kHzInput(spec: spec)
        return try prediction(input: input_, using: state)
    }
}
