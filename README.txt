RTX 5090 FP4 autotune input-preparation allocation validation

Source change: 5e8a5a1cf577fef81a58ba772c652ab71331c808
Target: flashinfer-ai/flashinfer, main base 0179e5b312d6aa0c790e8e571b9037bb277f4dbd

Measured environment: RTX 5090 SM120, FlashInfer 0.7.0.post1,
PyTorch 2.13.0+cu130, Triton 3.7.1. See model-validation.json.
The GPU tests applied the same targeted initializer change in an isolated
release-package overlay. This is not a claim that the entire current-main
package or every FP4 backend has been validated.

Memory experiment
  python probe_initializer_memory.py

The portable probe explicitly builds pre-change and post-change configs,
so it can be used from either source revision. It retains the same template
inputs, generates genuine AutoTuner optimization profiles, and measures
max_memory_allocated minus live template inputs while preparing each profile.
It writes initializer-memory-local.json. Recorded counters from the original
measurement are in initializer-memory.json; only output-path plumbing and
explicit baseline-config construction differ in the portable script.

For N=34816, K=5120:
  M=1:    320.108 -> 40.108 MiB extra preparation peak
  M=2048: 549.625 -> 181.625 MiB extra preparation peak
These are isolated profile-preparation peaks. They exclude later rotating
profiling batches, other model allocations, and whole-service peak memory.

Regression/correctness selection
  python -m pytest tests/gemm/test_mm_fp4.py::test_mm_fp4_autotune_profile_memory -q
  python -m pytest tests/gemm/test_mm_fp4.py::test_mm_fp4 -k cutlass -q

Both allocation checks fail at the intended assertion before the change.
With the change, the two allocation checks and two CUTLASS smoke cases pass.
The attached before/after logs use the repository test file with the pinned
release runtime; package-source paths have been replaced with $WORKSPACE.

Complete-model validation
The pinned unsloth/Qwen3.8-27B-NVFP4 model was loaded from byte-identical
resharded tensors in WSL. Two identical TTLCache requests were run per fresh
process, with a fresh FlashInfer autotune cache for each arm. See the issue
reference and common serving parameters in model-validation.json.
At utilization 0.92 both arms produced identical full text and 777 completion
tokens per request. At 0.91 texts differed (777 versus 767 tokens); all extracted
modules passed the same functional checks. Raw module-format checks failed in
all arms because the model included Markdown fences. The extraction-aware
validator is check_ttl.py. No code repairs were applied.
Both versions still fail at utilization 0.94. No general OOM fix, inference
throughput gain, or increased serving capacity is claimed.

Other architectures and native 8x4-layout backends were not tested locally.
Scripts and evidence in this branch are provided under Apache-2.0.
