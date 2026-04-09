#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to the CoreML model context.
typedef void* EveModelContext;

/// Load the CoreML model from a .mlpackage URL.
/// Returns an opaque context, or NULL on failure.
EveModelContext eve_model_load(const char* model_path);

/// Run a single inference frame.
/// input: pointer to kFrameSamples (960) float32 samples.
/// output: pointer to kHopSamples (480) float32 samples.
/// Returns 0 on success, non-zero on error.
int eve_model_predict(const float* input, float* output, void* context);

/// Release the model and all associated resources.
void eve_model_unload(EveModelContext context);

#ifdef __cplusplus
}
#endif
