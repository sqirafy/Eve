// CoreMLInference.mm — CoreML inference backend (dpdfnet2 16 kHz, FP16 weights)
// Audio pipeline: 48 kHz capture → 3:1 downsample → 16 kHz model → 3:1 upsample → 48 kHz output.
//
// Model I/O (DPDFNet2_16kHz.mlmodelc):
//   Input  "spec":     [1, 1, 161, 2]  — real/imag stacked spectrogram (16 kHz)
//   Input  "state_in": [45424]          — flattened GRU/DPRNN hidden state
//   Output "spec_e":   [1, 1, 161, 2]  — enhanced spectrogram
//   Output "state_out":[45424]          — updated state

#import <CoreML/CoreML.h>
#import <Accelerate/Accelerate.h>

#include "CoreMLInference.h"
#include "DPDFNetInitState.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants — 48 kHz capture side
// ---------------------------------------------------------------------------
static constexpr int kWinLen48  = 960;   // 20 ms @ 48 kHz
static constexpr int kHopSize48 = 480;   // 10 ms @ 48 kHz

// Constants — 16 kHz model side (3:1 decimation)
static constexpr int kSRCFactor = 3;
static constexpr int kWinLen16  = kWinLen48  / kSRCFactor;  // 320
static constexpr int kHopSize16 = kHopSize48 / kSRCFactor;  // 160
static constexpr int kFreqBins  = kWinLen16  / 2 + 1;       // 161

// ---------------------------------------------------------------------------
// 63-tap Hamming-windowed sinc lowpass, cutoff = 8 kHz / 48 kHz = 1/6
// ---------------------------------------------------------------------------
static constexpr int kSRCFilterLen = 63;

static const float kDownsampleFilter[kSRCFilterLen] = {
    0.00071139f, -0.00000000f, -0.00084996f, -0.00099530f,  0.00000000f,
    0.00146102f,  0.00179125f, -0.00000000f, -0.00267364f, -0.00323818f,
    0.00000000f,  0.00465070f,  0.00551674f, -0.00000000f, -0.00762684f,
   -0.00890222f,  0.00000000f,  0.01200428f,  0.01389555f, -0.00000000f,
   -0.01860930f, -0.02159335f,  0.00000000f,  0.02952783f,  0.03501036f,
   -0.00000000f, -0.05194594f, -0.06634666f,  0.00000000f,  0.13653375f,
    0.27501261f,  0.33333184f,  0.27501261f,  0.13653375f,  0.00000000f,
   -0.06634666f, -0.05194594f, -0.00000000f,  0.03501036f,  0.02952783f,
    0.00000000f, -0.02159335f, -0.01860930f, -0.00000000f,  0.01389555f,
    0.01200428f,  0.00000000f, -0.00890222f, -0.00762684f, -0.00000000f,
    0.00551674f,  0.00465070f,  0.00000000f, -0.00323818f, -0.00267364f,
   -0.00000000f,  0.00179125f,  0.00146102f,  0.00000000f, -0.00099530f,
   -0.00084996f, -0.00000000f,  0.00071139f
};

static const float kUpsampleFilter[kSRCFilterLen] = {
    0.00213417f, -0.00000000f, -0.00254987f, -0.00298590f,  0.00000000f,
    0.00438306f,  0.00537374f, -0.00000000f, -0.00802091f, -0.00971454f,
    0.00000000f,  0.01395209f,  0.01655022f, -0.00000000f, -0.02288053f,
   -0.02670667f,  0.00000000f,  0.03601285f,  0.04168665f, -0.00000000f,
   -0.05582791f, -0.06478004f,  0.00000000f,  0.08858349f,  0.10503108f,
   -0.00000000f, -0.15583783f, -0.19903999f,  0.00000000f,  0.40960125f,
    0.82503784f,  0.99999551f,  0.82503784f,  0.40960125f,  0.00000000f,
   -0.19903999f, -0.15583783f, -0.00000000f,  0.10503108f,  0.08858349f,
    0.00000000f, -0.06478004f, -0.05582791f, -0.00000000f,  0.04168665f,
    0.03601285f,  0.00000000f, -0.02670667f, -0.02288053f, -0.00000000f,
    0.01655022f,  0.01395209f,  0.00000000f, -0.00971454f, -0.00802091f,
   -0.00000000f,  0.00537374f,  0.00438306f,  0.00000000f, -0.00298590f,
   -0.00254987f, -0.00000000f,  0.00213417f
};

// ---------------------------------------------------------------------------
// Vorbis window
// ---------------------------------------------------------------------------
static void build_vorbis_window(float* w, int N) {
    const float inv_h = 1.0f / (N / 2.0f);
    for (int n = 0; n < N; ++n) {
        float s = std::sin(0.5f * float(M_PI) * (n + 0.5f) * inv_h);
        w[n] = std::sin(0.5f * float(M_PI) * s * s);
    }
}

// ---------------------------------------------------------------------------
// STFT/ISTFT using vDSP
// ---------------------------------------------------------------------------
struct DFTState {
    vDSP_DFT_Setup fwd = nullptr;
    vDSP_DFT_Setup inv = nullptr;
    explicit DFTState(int N) {
        fwd = vDSP_DFT_zop_CreateSetup(nullptr, N, vDSP_DFT_FORWARD);
        inv = vDSP_DFT_zop_CreateSetup(nullptr, N, vDSP_DFT_INVERSE);
        if (!fwd || !inv)
            throw std::runtime_error("vDSP_DFT setup failed");
    }
    ~DFTState() {
        if (fwd) vDSP_DFT_DestroySetup(fwd);
        if (inv) vDSP_DFT_DestroySetup(inv);
    }
    DFTState(const DFTState&) = delete;
    DFTState& operator=(const DFTState&) = delete;
};

static void rfft(const float* x, int N,
                 std::vector<float>& tmp_re, std::vector<float>& tmp_im,
                 std::vector<float>& out_re, std::vector<float>& out_im,
                 vDSP_DFT_Setup fwd_setup) {
    std::copy(x, x + N, tmp_re.begin());
    std::fill(tmp_im.begin(), tmp_im.end(), 0.0f);
    vDSP_DFT_Execute(fwd_setup, tmp_re.data(), tmp_im.data(),
                     out_re.data(), out_im.data());
}

static void irfft(const float* in_re, const float* in_im, int N, float* y,
                  std::vector<float>& tmp_re, std::vector<float>& tmp_im,
                  std::vector<float>& out_re, std::vector<float>& out_im,
                  vDSP_DFT_Setup inv_setup) {
    int half = N / 2;
    for (int k = 0; k <= half; ++k) { tmp_re[k] = in_re[k]; tmp_im[k] = in_im[k]; }
    for (int k = half + 1; k < N; ++k) { tmp_re[k] = in_re[N-k]; tmp_im[k] = -in_im[N-k]; }
    vDSP_DFT_Execute(inv_setup, tmp_re.data(), tmp_im.data(),
                     out_re.data(), out_im.data());
    float norm = 1.0f / float(N);
    vDSP_vsmul(out_re.data(), 1, &norm, y, 1, N);
}

// ---------------------------------------------------------------------------
// Reusable MLFeatureProvider
// ---------------------------------------------------------------------------
@interface EveFeatureProvider : NSObject <MLFeatureProvider>
- (instancetype)initWithSpec:(MLMultiArray*)spec stateIn:(MLMultiArray*)stateIn;
@end

@implementation EveFeatureProvider {
    MLMultiArray* _spec;
    MLMultiArray* _stateIn;
    NSSet<NSString*>* _names;
}
- (instancetype)initWithSpec:(MLMultiArray*)spec stateIn:(MLMultiArray*)stateIn {
    self = [super init];
    if (self) {
        _spec = spec; _stateIn = stateIn;
        _names = [NSSet setWithObjects:@"spec", @"state_in", nil];
    }
    return self;
}
- (NSSet<NSString*>*)featureNames { return _names; }
- (MLFeatureValue*)featureValueForName:(NSString*)name {
    if ([name isEqualToString:@"spec"])     return [MLFeatureValue featureValueWithMultiArray:_spec];
    if ([name isEqualToString:@"state_in"]) return [MLFeatureValue featureValueWithMultiArray:_stateIn];
    return nil;
}
@end

// ---------------------------------------------------------------------------
// Model context
// ---------------------------------------------------------------------------
struct EveModelContextImpl {
    MLModel*            model;
    EveFeatureProvider*  feature_provider;
    MLMultiArray*       spec_array;
    MLMultiArray*       state_in_array;

    std::vector<float> spec_buf;   // [kFreqBins * 2]
    std::vector<float> state;      // [kStateSize]

    // SRC history
    std::vector<float> ds_history;
    std::vector<float> us_history;

    // 16 kHz working buffers
    std::vector<float> frame16;
    std::vector<float> window16;
    std::vector<float> ola16;
    std::vector<float> hop16;

    DFTState dft16;
    std::vector<float> dft_re, dft_im;
    std::vector<float> idft_re, idft_im;
    std::vector<float> windowed16;
    std::vector<float> enh_re, enh_im;
    std::vector<float> time16;

    // SRC scratch buffers (pre-allocated, avoids per-frame heap alloc)
    std::vector<float> ds_scratch;  // [kSRCFilterLen - 1 + kWinLen48]
    std::vector<float> us_scratch;  // [kSRCFilterLen - 1 + kHopSize48]

    EveModelContextImpl(const char* model_path_utf8)
        : model(nil), feature_provider(nil), spec_array(nil), state_in_array(nil)
        , spec_buf(kFreqBins * 2, 0.0f)
        , state(kStateSize, 0.0f)
        , ds_history(kSRCFilterLen - 1, 0.0f)
        , us_history(kSRCFilterLen - 1, 0.0f)
        , frame16(kWinLen16, 0.0f)
        , window16(kWinLen16)
        , ola16(kWinLen16, 0.0f)
        , hop16(kHopSize16, 0.0f)
        , dft16(kWinLen16)
        , dft_re(kWinLen16), dft_im(kWinLen16)
        , idft_re(kWinLen16), idft_im(kWinLen16)
        , windowed16(kWinLen16)
        , enh_re(kFreqBins), enh_im(kFreqBins)
        , time16(kWinLen16)
        , ds_scratch(kSRCFilterLen - 1 + kWinLen48, 0.0f)
        , us_scratch(kSRCFilterLen - 1 + kHopSize48, 0.0f)
    {
        build_vorbis_window(window16.data(), kWinLen16);

        std::copy(kErbNormInit,  kErbNormInit  + kErbNormStateSize,  state.data());
        std::copy(kSpecNormInit, kSpecNormInit + kSpecNormStateSize,
                  state.data() + kErbNormStateSize);

        NSString* path = [NSString stringWithUTF8String:model_path_utf8];
        NSURL*    url  = [NSURL fileURLWithPath:path];

        MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = MLComputeUnitsAll;

        NSError* err = nil;
        model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
        if (!model) {
            NSString* msg = err ? err.localizedDescription : @"unknown error";
            throw std::runtime_error(std::string("CoreML load failed: ") + msg.UTF8String);
        }

        NSArray<NSNumber*>* spec_shape   = @[@1, @1, @(kFreqBins), @2];
        NSArray<NSNumber*>* spec_strides = @[@(kFreqBins * 2), @(kFreqBins * 2), @2, @1];
        spec_array = [[MLMultiArray alloc]
            initWithDataPointer:spec_buf.data() shape:spec_shape
                       dataType:MLMultiArrayDataTypeFloat32
                        strides:spec_strides deallocator:nil error:&err];
        if (!spec_array) throw std::runtime_error("MLMultiArray spec alloc failed");

        state_in_array = [[MLMultiArray alloc]
            initWithDataPointer:state.data() shape:@[@(kStateSize)]
                       dataType:MLMultiArrayDataTypeFloat32
                        strides:@[@1] deallocator:nil error:&err];
        if (!state_in_array) throw std::runtime_error("MLMultiArray state_in alloc failed");

        feature_provider = [[EveFeatureProvider alloc]
            initWithSpec:spec_array stateIn:state_in_array];
    }
};

// ---------------------------------------------------------------------------
// 3:1 Downsample using pre-allocated scratch buffer
// ---------------------------------------------------------------------------
static void downsample3(const float* in48, int n48,
                        std::vector<float>& history,
                        std::vector<float>& scratch,
                        float* out16, int n16) {
    const int tail = kSRCFilterLen - 1;
    std::copy(history.begin(), history.end(), scratch.begin());
    std::copy(in48, in48 + n48, scratch.begin() + tail);
    vDSP_desamp(scratch.data(), kSRCFactor, kDownsampleFilter, out16, n16, kSRCFilterLen);
    std::copy(in48 + n48 - tail, in48 + n48, history.begin());
}

// ---------------------------------------------------------------------------
// 3:1 Upsample using pre-allocated scratch buffer
// ---------------------------------------------------------------------------
static void upsample3(const float* in16, int n16,
                      std::vector<float>& history,
                      std::vector<float>& scratch,
                      float* out48, int n48) {
    const int tail = kSRCFilterLen - 1;
    std::copy(history.begin(), history.end(), scratch.begin());
    std::fill(scratch.begin() + tail, scratch.end(), 0.0f);
    for (int i = 0; i < n16; ++i)
        scratch[tail + i * kSRCFactor] = in16[i];
    vDSP_conv(scratch.data(), 1,
              kUpsampleFilter + kSRCFilterLen - 1, -1,
              out48, 1, n48, kSRCFilterLen);
    std::copy(scratch.end() - tail, scratch.end(), history.begin());
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------
EveModelContext eve_model_load(const char* model_path) {
    @autoreleasepool {
        try {
            return new EveModelContextImpl(model_path);
        } catch (const std::exception& e) {
            fprintf(stderr, "Eve: eve_model_load failed: %s\n", e.what());
            return nullptr;
        }
    }
}

int eve_model_predict(const float* input, float* output, void* context) {
    if (!context || !input || !output) return -1;
    auto* ctx = static_cast<EveModelContextImpl*>(context);

    @autoreleasepool {
        // 1. Downsample 48 kHz → 16 kHz
        downsample3(input, kWinLen48, ctx->ds_history, ctx->ds_scratch,
                    ctx->frame16.data(), kWinLen16);

        // 2. Analysis: vorbis window → rfft
        vDSP_vmul(ctx->frame16.data(), 1, ctx->window16.data(), 1,
                  ctx->windowed16.data(), 1, kWinLen16);

        rfft(ctx->windowed16.data(), kWinLen16,
             ctx->dft_re, ctx->dft_im, ctx->idft_re, ctx->idft_im,
             ctx->dft16.fwd);

        float* spec = ctx->spec_buf.data();
        for (int k = 0; k < kFreqBins; ++k) {
            spec[k * 2 + 0] = ctx->idft_re[k];
            spec[k * 2 + 1] = ctx->idft_im[k];
        }

        // 3. CoreML inference
        NSError* err = nil;
        id<MLFeatureProvider> result =
            [ctx->model predictionFromFeatures:ctx->feature_provider error:&err];
        if (!result) return -2;

        MLMultiArray* spec_e_arr    = [result featureValueForName:@"spec_e"].multiArrayValue;
        MLMultiArray* state_out_arr = [result featureValueForName:@"state_out"].multiArrayValue;
        if (!spec_e_arr || !state_out_arr) return -3;

        // FP16 models may return Float16 MLMultiArrays in release builds.
        // Use MLMultiArray subscript accessors which handle type conversion,
        // or convert manually when dataType is Float16.
        const int spec_e_count = kFreqBins * 2;
        float spec_e_f32[spec_e_count];
        if (spec_e_arr.dataType == MLMultiArrayDataTypeFloat16) {
            const __fp16* p = static_cast<const __fp16*>(spec_e_arr.dataPointer);
            for (int i = 0; i < spec_e_count; ++i) spec_e_f32[i] = static_cast<float>(p[i]);
        } else {
            std::memcpy(spec_e_f32, spec_e_arr.dataPointer, spec_e_count * sizeof(float));
        }
        const float* spec_e = spec_e_f32;

        if (state_out_arr.dataType == MLMultiArrayDataTypeFloat16) {
            const __fp16* p = static_cast<const __fp16*>(state_out_arr.dataPointer);
            for (int i = 0; i < kStateSize; ++i) ctx->state[i] = static_cast<float>(p[i]);
        } else {
            const float* state_out_ptr = static_cast<const float*>(state_out_arr.dataPointer);
            std::memcpy(ctx->state.data(), state_out_ptr, kStateSize * sizeof(float));
        }

        // 4. Synthesis: irfft → synthesis window → OLA
        for (int k = 0; k < kFreqBins; ++k) {
            ctx->enh_re[k] = spec_e[k * 2 + 0];
            ctx->enh_im[k] = spec_e[k * 2 + 1];
        }

        irfft(ctx->enh_re.data(), ctx->enh_im.data(), kWinLen16,
              ctx->time16.data(),
              ctx->dft_re, ctx->dft_im, ctx->idft_re, ctx->idft_im,
              ctx->dft16.inv);

        vDSP_vmul(ctx->time16.data(), 1, ctx->window16.data(), 1,
                  ctx->time16.data(), 1, kWinLen16);

        vDSP_vadd(ctx->ola16.data(), 1, ctx->time16.data(), 1,
                  ctx->ola16.data(), 1, kWinLen16);

        std::copy(ctx->ola16.begin(), ctx->ola16.begin() + kHopSize16,
                  ctx->hop16.begin());
        std::copy(ctx->ola16.begin() + kHopSize16, ctx->ola16.end(),
                  ctx->ola16.begin());
        std::fill(ctx->ola16.end() - kHopSize16, ctx->ola16.end(), 0.0f);

        // 5. Upsample 16 kHz → 48 kHz
        upsample3(ctx->hop16.data(), kHopSize16, ctx->us_history, ctx->us_scratch,
                  output, kHopSize48);

        return 0;
    }
}

void eve_model_unload(EveModelContext context) {
    if (!context) return;
    delete static_cast<EveModelContextImpl*>(context);
}
