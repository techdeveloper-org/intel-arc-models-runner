import json

index_path = r"C:\Users\techd\Documents\workspace-spring-tool-suite-4-4.27.0-new\claude-global-library\knowledge-graph\_discovery\index\agents_index.json"
with open(index_path, "r", encoding="utf-8") as f:
    data = json.load(f)

records = data.get("records", [])

categories = {
    "dsa_algorithms": ["algorithm", "dsa", "data-structure", "tree", "graph", "ast", "parser", "dag"],
    "systems_lowlevel": ["systems", "c-cpp", "memory", "compiler", "hardware", "accelerator", "kernel"],
    "design_architecture_solid": ["architect", "design", "solid", "pattern", "clean-code", "refactor", "oop", "software-design"],
    "exception_error_handling": ["exception", "error", "diagnostic", "safety", "resilience", "fault"],
    "optimization_performance": ["optimization", "performance", "benchmark", "profiler", "efficiency", "latency", "throughput"],
    "testing_qa": ["test", "qa", "quality", "verification", "validation", "harness", "regression", "fuzz", "negative"]
}

found = {k: [] for k in categories}

for r in records:
    name = r.get("name", "").lower()
    desc = r.get("description", "").lower()
    dom = r.get("domain", "").lower()
    text = f"{name} {desc} {dom}"
    for cat, keywords in categories.items():
        if any(kw in text for kw in keywords):
            found[cat].append({
                "name": r.get("name"),
                "domain": r.get("domain"),
                "path": r.get("path"),
                "desc": r.get("description", "")[:140]
            })

for cat, agents in found.items():
    print(f"\n=======================================================")
    print(f"=== {cat.upper()} (Total Found: {len(agents)}) ===")
    print(f"=======================================================")
    for a in agents[:15]:
        print(f"  * {a['name']} | Domain: {a['domain']}")
        if a['desc']:
            print(f"    Desc: {a['desc']}...")
