"""NPU draft 1-token decode step time via openvino_genai StaticLLMPipeline perf metrics."""
import json, os, time
import numpy as np
import openvino_genai as g
import psutil
R = "C:/Users/techd/Documents/workspace-spring-tool-suite-4-4.27.0-new/intel-arc-models-runner"
rss = lambda: psutil.Process(os.getpid()).memory_info().rss / 2**20
t0 = time.perf_counter()
pipe = g.LLMPipeline(R + "/models/qwen-0.5b-npu-certified", "NPU")
print(f"compile {time.perf_counter()-t0:.1f}s rss={rss():.0f}MB", flush=True)
cfg = g.GenerationConfig(); cfg.max_new_tokens = 24; cfg.do_sample = False
tpots, ttfts, pooled = [], [], []
for rep in range(53):
    r = pipe.generate(["Once upon a time there was a small village near the river"], cfg)
    pm = r.perf_metrics
    if rep >= 3:
        tpots.append(pm.get_tpot().mean); ttfts.append(pm.get_ttft().mean)
        try:
            pooled.extend(list(pm.raw_metrics.token_infer_durations))
        except Exception as e:
            pass
    if rep % 10 == 0:
        print(f"rep={rep} tpot={pm.get_tpot().mean:.2f}ms ttft={pm.get_ttft().mean:.2f}ms rss={rss():.0f}MB", flush=True)
a = np.array(tpots)
out = dict(n=len(a), tpot_p50=float(np.percentile(a, 50)), tpot_p95=float(np.percentile(a, 95)), tpot_mean=float(a.mean()),
           ttft_p50=float(np.percentile(ttfts, 50)), raw_infer_n=len(pooled))
if pooled:
    p = np.array(pooled); out.update(raw_p50=float(np.percentile(p, 50)), raw_p95=float(np.percentile(p, 95)), raw_unit_note="as reported by raw_metrics.token_infer_durations")
print(out)
json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + "/step_NPU.json", "w"), indent=1)
