#!/usr/bin/env python3
"""
Convert DPDFNet PyTorch checkpoint to CoreML (.mlpackage).

Pipeline:
  1. Instantiate streaming DPDFNet from DPDFNet/onnx_model/ (frame-by-frame
     with explicit state I/O: spec [1,1,F,2] + state [S] → spec_e + state_out).
  2. Load .pth checkpoint via correct_state_dict() weight remapping.
  3. Convert grouped linear layers to einsum for efficient tracing.
  4. Trace with torch.jit.trace.
  5. Convert to CoreML MLProgram via coremltools (FP16 compute precision).

Usage:
    python scripts/convert_model.py --model dpdfnet2 --output Eve/Model

Requirements (already in .venv):
    coremltools>=9, torch, einops
"""

import argparse
import os
import sys
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Add DPDFNet repo root to sys.path so onnx_model/ and model/ resolve as packages.
dpdfnet_repo = os.path.join(project_root, "DPDFNet")
if dpdfnet_repo not in sys.path:
    sys.path.insert(0, dpdfnet_repo)


def convert(model_name: str, output_dir: str, dprnn_num_blocks: int):
    import torch
    import coremltools as ct

    from onnx_model.dpdfnet import DPDFNet, correct_state_dict
    from onnx_model.layers import convert_grouped_linear_to_einsum

    # ------------------------------------------------------------------
    # Step 1: Resolve checkpoint
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(dpdfnet_repo, "model_zoo", "checkpoints", f"{model_name}.pth")
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
    stream_state_dict = correct_state_dict(state_dict)
    model.load_state_dict(stream_state_dict, strict=True)
    print("  weights loaded OK.")

    convert_grouped_linear_to_einsum(model)
    model.eval()

    freq_bins = model.freq_bins
    state_size = model.state_size()
    print(f"  freq_bins={freq_bins}  state_size={state_size}")

    # ------------------------------------------------------------------
    # Step 3: Wrap with wnorm scaling (same as ONNX export wrapper)
    # ------------------------------------------------------------------
    class DPDFNetCoreMLWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m
            self.register_buffer("wnorm", torch.tensor(float(m.wnorm), dtype=torch.float32))
            self.register_buffer("inv_wnorm", torch.tensor(1.0 / float(m.wnorm), dtype=torch.float32))

        def forward(self, spec: torch.Tensor, state_in: torch.Tensor):
            spec = spec * self.wnorm
            spec_e, state_out = self.model(spec, state_in)
            spec_e = spec_e * self.inv_wnorm
            return spec_e, state_out

    wrapper = DPDFNetCoreMLWrapper(model).eval()

    # ------------------------------------------------------------------
    # Step 4: Trace
    # ------------------------------------------------------------------
    spec_shape = [1, 1, freq_bins, 2]
    state_shape = [state_size]

    print(f"Tracing … spec={spec_shape}  state={state_shape}")
    spec_in = torch.zeros(*spec_shape)
    state_in = model.initial_state(dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (spec_in, state_in), strict=False)
    print("  tracing succeeded.")

    # ------------------------------------------------------------------
    # Step 5: CoreML conversion
    # ------------------------------------------------------------------
    print("Converting to CoreML MLProgram (FP16 compute, macOS 15) …")

    inputs = [
        ct.TensorType(name="spec",     shape=tuple(spec_shape),  dtype=np.float32),
        ct.TensorType(name="state_in", shape=tuple(state_shape), dtype=np.float32),
    ]
    outputs = [
        ct.TensorType(name="spec_e",    dtype=np.float32),
        ct.TensorType(name="state_out", dtype=np.float32),
    ]

    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )

    # ------------------------------------------------------------------
    # Step 6: Save
    # ------------------------------------------------------------------
    out_dir = os.path.join(project_root, output_dir)
    os.makedirs(out_dir, exist_ok=True)
    if "48khz" in model_name:
        output_name = "DPDFNet2_48kHz"
    else:
        output_name = "DPDFNet2_16kHz"
    output_path = os.path.join(out_dir, f"{output_name}.mlpackage")
    mlmodel.save(output_path)
    print(f"  saved → {output_path}")

    # ------------------------------------------------------------------
    # Step 7: Sanity check — PyTorch vs CoreML
    # ------------------------------------------------------------------
    print("Running sanity check (PyTorch vs CoreML) …")
    rng = np.random.default_rng(42)
    spec_np = rng.standard_normal(tuple(spec_shape)).astype(np.float32)
    state_np = model.initial_state(dtype=torch.float32).numpy()

    with torch.no_grad():
        pt_spec_e, pt_state_out = wrapper(
            torch.from_numpy(spec_np),
            torch.from_numpy(state_np),
        )
    pt_spec_e = pt_spec_e.numpy()

    cml_out = mlmodel.predict({"spec": spec_np, "state_in": state_np})
    cml_spec_e = cml_out["spec_e"]

    mae = float(np.mean(np.abs(pt_spec_e - cml_spec_e)))
    max_ae = float(np.max(np.abs(pt_spec_e - cml_spec_e)))
    print(f"  spec_e MAE={mae:.6f}  max_AE={max_ae:.6f}")
    if max_ae > 0.01:
        print("  WARNING: large numeric deviation")
    else:
        print("  Numeric match OK")

    print(f"\nDone.  CoreML model: {output_path}")
    print(f"  freq_bins={freq_bins}  state_size={state_size}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert DPDFNet PyTorch → CoreML")
    parser.add_argument("--model", default="dpdfnet2",
                        help="Checkpoint name in DPDFNet/model_zoo/checkpoints/ (default: dpdfnet2)")
    parser.add_argument("--output", default="Eve/Model",
                        help="Output directory for .mlpackage (default: Eve/Model)")
    parser.add_argument("--dprnn-num-blocks", type=int, default=2,
                        help="Number of DPRNN blocks (default: 2 for dpdfnet2)")
    args = parser.parse_args()
    convert(args.model, args.output, args.dprnn_num_blocks)


if __name__ == "__main__":
    main()
