#!/usr/bin/env python3
"""
Convert DPDFNet PyTorch checkpoint → stateful, palettized CoreML MLProgram.

Architectural improvements over the original pipeline:

  1. STATEFUL MODEL (biggest win)
     The original model exposed GRU/DPRNN hidden state as explicit tensor
     inputs/outputs (state_in [45424] → state_out [45424]). On every inference
     call this forced a 45 kB CPU↔ANE memory transfer. The stateful model uses
     CoreML MLState so the state lives inside the ANE memory subsystem and is
     updated in-place — zero copies per frame.
     Initial state (ERB norm + spec norm) is embedded in the model buffer so
     runtime initialization code (DPDFNetInitState.h) is no longer needed.

  2. 4-BIT WEIGHT PALETTIZATION
     The ANE has dedicated hardware for decompressing palettized (lookup-table)
     weights on-chip. Converting FP16 weights to 4-bit palettized form reduces
     weight DRAM bandwidth by ~4x, lowering power and improving throughput.

  3. CPU_AND_NE COMPUTE UNITS
     For a streaming model with batch size 1 and sequential GRU ops, the GPU
     adds scheduling overhead without throughput benefit. CPU_AND_NE routes
     all matmul/conv ops to the ANE and handles residual ops on CPU.

Usage:
    python scripts/convert_model.py --model dpdfnet2 --output Eve/Model

Requirements (already in .venv):
    coremltools>=7, torch, einops
"""

import argparse
import os
import sys
import numpy as np

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

dpdfnet_repo = os.path.join(project_root, "DPDFNet")
if dpdfnet_repo not in sys.path:
    sys.path.insert(0, dpdfnet_repo)


def convert(model_name: str, output_dir: str, dprnn_num_blocks: int,
            palettize: bool = True):
    import torch
    import coremltools as ct
    import coremltools.optimize.coreml as cto

    from onnx_model.dpdfnet import DPDFNet, correct_state_dict
    from onnx_model.layers import convert_grouped_linear_to_einsum

    # ------------------------------------------------------------------
    # Step 1: Resolve checkpoint
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(
        dpdfnet_repo, "model_zoo", "checkpoints", f"{model_name}.pth")
    if not os.path.exists(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")
    print(f"Checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Step 2: Build streaming model and load weights
    # ------------------------------------------------------------------
    print("Building streaming DPDFNet model …")
    model = DPDFNet(
        conv_kernel_inp=(3, 3),
        conv_ch=64,
        enc_gru_dim=256,
        erb_dec_gru_dim=256,
        df_dec_gru_dim=256,
        enc_lin_groups=32,
        lin_groups=16,
        upsample_conv_type="subpixel",
        group_linear_type="loop",
        point_wise_type="cnn",
        separable_first_conv=True,
        dprnn_num_blocks=dprnn_num_blocks,
    )

    print("Loading checkpoint …")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(correct_state_dict(state_dict), strict=True)
    print("  weights loaded OK.")

    convert_grouped_linear_to_einsum(model)
    model.eval()

    freq_bins  = model.freq_bins
    state_size = model.state_size()
    print(f"  freq_bins={freq_bins}  state_size={state_size}")

    # ------------------------------------------------------------------
    # Step 3: Stateful wrapper
    #
    # DPDFNetStatefulWrapper replaces the original DPDFNetCoreMLWrapper.
    # Key difference: state is a registered buffer rather than an explicit
    # tensor input/output. coremltools recognises the buffer + copy_() pattern
    # and converts it to read_state / coreml_update_state MIL ops so that
    # CoreML manages the state tensor inside MLState — no CPU copies needed.
    #
    # The buffer is initialised with model.initial_state() which already
    # contains the ERB norm and spec norm initial values, so the runtime
    # DPDFNetInitState.h constants are no longer needed.
    # ------------------------------------------------------------------
    class DPDFNetStatefulWrapper(torch.nn.Module):
        def __init__(self, m: torch.nn.Module, initial_state: torch.Tensor):
            super().__init__()
            self.net      = m
            self.register_buffer(
                "wnorm",     torch.tensor(float(m.wnorm),        dtype=torch.float32))
            self.register_buffer(
                "inv_wnorm", torch.tensor(1.0 / float(m.wnorm), dtype=torch.float32))
            # Stored as FP16 — CoreML MLState requires fp16 dtype.
            # Name "gru_state" avoids the CoreML reserved name "state".
            self.register_buffer("gru_state", initial_state.clone().half())

        def forward(self, spec: torch.Tensor) -> torch.Tensor:
            # Cast buffer FP16→FP32 for the model, then store result as FP16.
            spec_e, new_state = self.net(spec * self.wnorm, self.gru_state.float())
            # Slice assignment generates aten::slice + aten::copy_ in the JIT IR,
            # which coremltools maps to coreml_update_state MIL op.
            self.gru_state[:] = new_state.half()
            return spec_e * self.inv_wnorm

    initial_state = model.initial_state(dtype=torch.float32)
    wrapper = DPDFNetStatefulWrapper(model, initial_state).eval()

    # ------------------------------------------------------------------
    # Step 4: Trace (stateful — spec only, state managed by buffer)
    # ------------------------------------------------------------------
    spec_shape = (1, 1, freq_bins, 2)
    spec_in    = torch.zeros(*spec_shape)

    print(f"Tracing … spec={list(spec_shape)}")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (spec_in,), strict=False)
    print("  tracing succeeded.")

    # ------------------------------------------------------------------
    # Step 5: CoreML conversion — stateful, CPU_AND_NE, FP16
    # ------------------------------------------------------------------
    print("Converting to stateful CoreML MLProgram "
          "(FP16 compute, CPU_AND_NE, macOS 15) …")

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="spec", shape=spec_shape, dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="spec_e", dtype=np.float32),
        ],
        states=[
            # Declares 'gru_state' as an MLState variable (fp16 — required by CoreML).
            # CoreML initialises it with the buffer's embedded values
            # (ERB norm + spec norm + GRU zeros, quantised to fp16).
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(state_size,), dtype=np.float16),
                name="gru_state",
            ),
        ],
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        # CPU_AND_NE: optimal for M-series streaming models (batch=1, sequential
        # GRU ops). Avoids GPU scheduling overhead; ANE handles matmul/conv.
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print("  CoreML conversion succeeded.")

    # ------------------------------------------------------------------
    # Step 6: 4-bit weight palettization
    #
    # The ANE has on-chip hardware for decompressing palettized weights,
    # so this reduces DRAM bandwidth by ~4x with minimal accuracy loss.
    # We skip weights smaller than 2048 elements (overhead exceeds benefit).
    # ------------------------------------------------------------------
    if palettize:
        print("Palettizing weights (4-bit k-means) …")
        config = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                nbits=4,
                mode="kmeans",
                weight_threshold=2048,
            )
        )
        mlmodel = cto.palettize_weights(mlmodel, config)
        print("  palettization done.")

    # ------------------------------------------------------------------
    # Step 7: Save
    # ------------------------------------------------------------------
    out_dir = os.path.join(project_root, output_dir)
    os.makedirs(out_dir, exist_ok=True)
    output_name = "DPDFNet2_48kHz" if "48khz" in model_name else "DPDFNet2_16kHz"
    output_path = os.path.join(out_dir, f"{output_name}.mlpackage")
    mlmodel.save(output_path)
    print(f"  saved → {output_path}")

    # ------------------------------------------------------------------
    # Step 8: Sanity check — PyTorch first frame vs CoreML first frame
    #
    # Both start from the same initial state (embedded in the model buffer /
    # CoreML MLState default), so the outputs should match closely.
    # ------------------------------------------------------------------
    print("Running sanity check (PyTorch vs CoreML first frame) …")
    rng     = np.random.default_rng(42)
    spec_np = rng.standard_normal(spec_shape).astype(np.float32)

    # Fresh PyTorch wrapper (state reset to initial values).
    ref_wrapper = DPDFNetStatefulWrapper(model, initial_state).eval()
    with torch.no_grad():
        pt_spec_e = ref_wrapper(torch.from_numpy(spec_np)).numpy()

    # CoreML stateful prediction — create a fresh state and pass it explicitly.
    cml_state  = mlmodel.make_state()
    cml_out    = mlmodel.predict({"spec": spec_np}, state=cml_state)
    cml_spec_e = cml_out["spec_e"]

    mae    = float(np.mean(np.abs(pt_spec_e - cml_spec_e)))
    max_ae = float(np.max(np.abs(pt_spec_e  - cml_spec_e)))
    print(f"  spec_e MAE={mae:.6f}  max_AE={max_ae:.6f}")
    # Palettization introduces small quantisation error; allow up to 0.05.
    threshold = 0.05 if palettize else 0.01
    if max_ae > threshold:
        print(f"  WARNING: numeric deviation exceeds {threshold:.2f} — "
              f"consider --no-palettize")
    else:
        print("  Numeric match OK")

    print(f"\nDone. CoreML model: {output_path}")
    print(f"  freq_bins={freq_bins}  state_size={state_size}")
    print(f"  stateful=True  palettized={palettize}  compute=CPU_AND_NE")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert DPDFNet PyTorch → stateful CoreML MLProgram")
    parser.add_argument("--model", default="dpdfnet2",
        help="Checkpoint name in DPDFNet/model_zoo/checkpoints/")
    parser.add_argument("--output", default="Eve/Model",
        help="Output directory for .mlpackage")
    parser.add_argument("--dprnn-num-blocks", type=int, default=2,
        help="Number of DPRNN blocks (0=baseline, 2=dpdfnet2)")
    parser.add_argument("--no-palettize", action="store_true",
        help="Skip 4-bit weight palettization")
    args = parser.parse_args()
    convert(args.model, args.output, args.dprnn_num_blocks,
            palettize=not args.no_palettize)


if __name__ == "__main__":
    main()
