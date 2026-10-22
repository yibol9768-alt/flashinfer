"""Measure genuine FP4 profile input synthesis before and after scratch init changes."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import torch
from flashinfer.autotuner import AutoTuner
from flashinfer.autotuner.initializers import autotuner_initializer_empty
from flashinfer.gemm.gemm_base import _MM_FP4_TUNING_CONFIG_128x4, DEFAULT_WORKSPACE_SIZE

LAB = Path(__file__).resolve().parent
device = torch.device('cuda', 0)
m, n, k = 2048, 34816, 5120
inputs = [torch.empty((m,k//2),dtype=torch.uint8,device=device),
          torch.empty((n,k//2),dtype=torch.uint8,device=device).t(),
          torch.empty((m,k//16),dtype=torch.uint8,device=device),
          torch.empty((n,k//16),dtype=torch.uint8,device=device).t(),
          torch.ones((),dtype=torch.float32,device=device),torch.bfloat16,
          torch.empty((m,n),dtype=torch.bfloat16,device=device),16,True,
          torch.empty(DEFAULT_WORKSPACE_SIZE,dtype=torch.uint8,device=device)]
tuner = AutoTuner.get()
rows = []
baseline_config = replace(_MM_FP4_TUNING_CONFIG_128x4, tensor_initializers=())
for label, config in [('original',baseline_config),
                      ('empty_output_and_workspace',replace(baseline_config,tensor_initializers=((6,autotuner_initializer_empty),(9,autotuner_initializer_empty))))]:
    profiles = tuner._generate_optimization_profiles(config, inputs)
    for profile in profiles:
        shape_m = profile.get_opt_shapes()[0][0]
        if shape_m not in (1,256,2048):
            continue
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        tensors = tuner._prepare_input_tensors(profile, inputs)
        torch.cuda.synchronize()
        row={'config':label,'m':shape_m,'peak_extra_bytes':torch.cuda.max_memory_allocated()-before,
             'live_extra_bytes':torch.cuda.memory_allocated()-before,
             'output_shape':list(tensors[6].shape),'workspace_bytes':tensors[9].numel(),
             'static_weights_reused':tensors[1] is inputs[1]}
        rows.append(row)
        print(json.dumps(row),flush=True)
        del tensors
(LAB/'initializer-memory-local.json').write_text(json.dumps({'rows':rows},indent=2)+'\n')
