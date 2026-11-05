import json, sys, glob
rows = {"top_k=20": 0.20, "top_k=50 top_p=0.9": 0.20, "top_p=0.9 only": 2.8, "plain softmax": 1.9}
tgt = {}
for f in glob.glob("step_GPU*.json") + glob.glob("step_CPU*.json"):
    d = json.load(open(f)); dev = f.split("_")[1].split(".")[0].split("_")[0]
    tgt.setdefault(dev, {}).update(d)
npu = json.load(open("step_NPU.json"))["tpot_p50"] if len(glob.glob("step_NPU.json")) else None
print("NPU draft step p50 ms:", npu)
for dev, d in sorted(tgt.items()):
    for P in (5, 100):
        k = f"P{P}_c4"
        if k not in d: print(dev, P, "not run"); continue
        for stat in ("p50", "p95"):
            cyc = d[k][stat] + (3 * npu if npu else 0)
            line = []
            for name, r in rows.items():
                s = 7 * r + 0.5
                line.append(f"{name}: s={s:.1f}ms ov={100*s/cyc:.2f}%{'' if 100*s/cyc<5 else ' FAIL'}")
            print(f"{dev} P={P} {stat} target4={d[k][stat]:.1f} cycle(target4+3*draft)={cyc:.1f} | " + " | ".join(line))
