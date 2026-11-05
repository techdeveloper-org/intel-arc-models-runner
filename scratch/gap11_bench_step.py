"""Time one infer of a 4-token / 1-token chunk on a plain OpenVINO compiled model."""
import argparse, json, os, sys, time
import numpy as np
import openvino as ov

R = "C:/Users/techd/Documents/workspace-spring-tool-suite-4-4.27.0-new/intel-arc-models-runner"
MODEL = R + "/models/qwen-2.5-1.5b/openvino_model.xml"
CFG = {
    "GPU": {"KV_CACHE_PRECISION": "f16"},
    "CPU": {"INFERENCE_PRECISION_HINT": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0, "KV_CACHE_PRECISION": "f32"},
}

def rss():
    import psutil
    return psutil.Process(os.getpid()).memory_info().rss / 2**20

def feed(req, toks, start):
    n = len(toks)
    req.infer({"input_ids": np.array([toks], dtype=np.int64),
               "attention_mask": np.ones((1, start + n), dtype=np.int64),
               "position_ids": np.arange(start, start + n, dtype=np.int64)[None, :],
               "beam_idx": np.array([0], dtype=np.int32)})
    return np.array(req.get_tensor("logits").data)[0, -1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--P", type=int, nargs="+", default=[5, 100])
    a = ap.parse_args()
    t0 = time.perf_counter()
    model = ov.Core().compile_model(MODEL, a.dev, CFG[a.dev])
    print(f"compile {time.perf_counter()-t0:.1f}s rss={rss():.0f}MB", flush=True)
    req = model.create_infer_request()
    rng = np.random.default_rng(0)
    out = {}
    for P in a.P:
        prompt = [int(x) for x in rng.integers(1000, 50000, size=P)]
        for chunk in (4, 1):
            ts = []
            for rep in range(a.reps + 3):
                req.reset_state()
                feed(req, prompt, 0)
                toks = [int(x) for x in rng.integers(1000, 50000, size=chunk)]
                t = time.perf_counter()
                feed(req, toks, P)
                dt = (time.perf_counter() - t) * 1000
                if rep >= 3:
                    ts.append(dt)
                if rep % 10 == 0:
                    print(f"{a.dev} P={P} chunk={chunk} rep={rep} {dt:.2f}ms rss={rss():.0f}MB", flush=True)
            arr = np.array(ts)
            out[f"P{P}_c{chunk}"] = dict(n=len(ts), p50=float(np.percentile(arr, 50)), p95=float(np.percentile(arr, 95)),
                                         mean=float(arr.mean()), min=float(arr.min()), max=float(arr.max()))
            print(a.dev, P, chunk, out[f"P{P}_c{chunk}"], flush=True)
    json.dump(out, open(os.path.dirname(os.path.abspath(__file__)) + f"/step_{a.dev}_P{"_".join(map(str,a.P))}.json", "w"), indent=1)
main()
