//
// DPDFNet2_48kHz.swift
//
// This file was automatically generated and should not be edited.
//

import CoreML


/// Model Prediction Input Type
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
class DPDFNet2_48kHzInput : MLFeatureProvider {

    /// spec as 1 × 1 × 481 × 2 4-dimensional array of floats
    var spec: MLMultiArray

    /// state_in as 56436 element vector of floats
    var state_in: MLMultiArray

    var featureNames: Set<String> { ["spec", "state_in"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "spec" {
            return MLFeatureValue(multiArray: spec)
        }
        if featureName == "state_in" {
            return MLFeatureValue(multiArray: state_in)
        }
        return nil
    }

    init(spec: MLMultiArray, state_in: MLMultiArray) {
        self.spec = spec
        self.state_in = state_in
    }

    convenience init(spec: MLShapedArray<Float>, state_in: MLShapedArray<Float>) {
        self.init(spec: MLMultiArray(spec), state_in: MLMultiArray(state_in))
    }

}


/// Model Prediction Output Type
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
class DPDFNet2_48kHzOutput : MLFeatureProvider {

    /// Source provided by CoreML
    private let provider : MLFeatureProvider

    /// spec_e as 1 × 1 × 481 × 2 4-dimensional array of floats
    var spec_e: MLMultiArray {
        provider.featureValue(for: "spec_e")!.multiArrayValue!
    }

    /// spec_e as 1 × 1 × 481 × 2 4-dimensional array of floats
    var spec_eShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(spec_e)
    }

    /// state_out as 56436 element vector of floats
    var state_out: MLMultiArray {
        provider.featureValue(for: "state_out")!.multiArrayValue!
    }

    /// state_out as 56436 element vector of floats
    var state_outShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(state_out)
    }

    var featureNames: Set<String> {
        provider.featureNames
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        provider.featureValue(for: featureName)
    }

    init(spec_e: MLMultiArray, state_out: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["spec_e" : MLFeatureValue(multiArray: spec_e), "state_out" : MLFeatureValue(multiArray: state_out)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}


/// Class for model loading and prediction
@available(macOS 15.0, iOS 18.0, tvOS 18.0, watchOS 11.0, visionOS 2.0, *)
class DPDFNet2_48kHz {
    let model: MLModel

    /// URL of model assuming it was installed in the same bundle as this class
    class var urlOfModelInThisBundle : URL {
        let bundle = Bundle(for: self)
        return bundle.url(forResource: "DPDFNet2_48kHz", withExtension:"mlmodelc")!
    }

    /**
        Construct DPDFNet2_48kHz instance with an existing MLModel object.

        Usually the application does not use this initializer unless it makes a subclass of DPDFNet2_48kHz.
        Such application may want to use `MLModel(contentsOfURL:configuration:)` and `DPDFNet2_48kHz.urlOfModelInThisBundle` to create a MLModel object to pass-in.

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
        Construct DPDFNet2_48kHz instance with explicit path to mlmodelc file
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
        Construct DPDFNet2_48kHz instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<DPDFNet2_48kHz, Error>) -> Void) {
        load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration, completionHandler: handler)
    }

    /**
        Construct DPDFNet2_48kHz instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
    */
    class func load(configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> DPDFNet2_48kHz {
        try await load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct DPDFNet2_48kHz instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<DPDFNet2_48kHz, Error>) -> Void) {
        MLModel.load(contentsOf: modelURL, configuration: configuration) { result in
            switch result {
            case .failure(let error):
                handler(.failure(error))
            case .success(let model):
                handler(.success(DPDFNet2_48kHz(model: model)))
            }
        }
    }

    /**
        Construct DPDFNet2_48kHz instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
    */
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> DPDFNet2_48kHz {
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        return DPDFNet2_48kHz(model: model)
    }

    /**
        Make a prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as DPDFNet2_48kHzInput

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as DPDFNet2_48kHzOutput
    */
    func prediction(input: DPDFNet2_48kHzInput) throws -> DPDFNet2_48kHzOutput {
        try prediction(input: input, options: MLPredictionOptions())
    }

    /**
        Make a prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as DPDFNet2_48kHzInput
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as DPDFNet2_48kHzOutput
    */
    func prediction(input: DPDFNet2_48kHzInput, options: MLPredictionOptions) throws -> DPDFNet2_48kHzOutput {
        let outFeatures = try model.prediction(from: input, options: options)
        return DPDFNet2_48kHzOutput(features: outFeatures)
    }

    /**
        Make an asynchronous prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as DPDFNet2_48kHzInput
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as DPDFNet2_48kHzOutput
    */
    func prediction(input: DPDFNet2_48kHzInput, options: MLPredictionOptions = MLPredictionOptions()) async throws -> DPDFNet2_48kHzOutput {
        let outFeatures = try await model.prediction(from: input, options: options)
        return DPDFNet2_48kHzOutput(features: outFeatures)
    }

    /**
        Make a prediction using the convenience interface

        It uses the default function if the model has multiple functions.

        - parameters:
            - spec: 1 × 1 × 481 × 2 4-dimensional array of floats
            - state_in: 56436 element vector of floats

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as DPDFNet2_48kHzOutput
    */
    func prediction(spec: MLMultiArray, state_in: MLMultiArray) throws -> DPDFNet2_48kHzOutput {
        let input_ = DPDFNet2_48kHzInput(spec: spec, state_in: state_in)
        return try prediction(input: input_)
    }

    /**
        Make a prediction using the convenience interface

        It uses the default function if the model has multiple functions.

        - parameters:
            - spec: 1 × 1 × 481 × 2 4-dimensional array of floats
            - state_in: 56436 element vector of floats

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as DPDFNet2_48kHzOutput
    */

    func prediction(spec: MLShapedArray<Float>, state_in: MLShapedArray<Float>) throws -> DPDFNet2_48kHzOutput {
        let input_ = DPDFNet2_48kHzInput(spec: spec, state_in: state_in)
        return try prediction(input: input_)
    }

    /**
        Make a batch prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - inputs: the inputs to the prediction as [DPDFNet2_48kHzInput]
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as [DPDFNet2_48kHzOutput]
    */
    func predictions(inputs: [DPDFNet2_48kHzInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [DPDFNet2_48kHzOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [DPDFNet2_48kHzOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  DPDFNet2_48kHzOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
}
