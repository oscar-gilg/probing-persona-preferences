"""Add unpadded-layer probe aliases to layer_sweep probe manifests.

The runner builds probe_id via f"{probe_prefix}{layer}" — e.g. ridge_L2 for layer 2.
Our manifests were saved with zero-padded names (ridge_L02). Adding duplicate
entries with the unpadded id so both lookups succeed.
"""
import json
from pathlib import Path

for sel in ["tb-2", "eot"]:
    p = Path(f"results/probes/layer_sweep/{sel}/manifest.json")
    m = json.loads(p.read_text())
    existing_ids = {e["id"] for e in m["probes"]}
    new_entries = []
    for e in m["probes"]:
        if e["id"].startswith("ridge_L"):
            layer = e["layer"]
            unpadded = f"ridge_L{layer}"
            if unpadded not in existing_ids:
                alias = dict(e)
                alias["id"] = unpadded
                new_entries.append(alias)
    m["probes"].extend(new_entries)
    p.write_text(json.dumps(m, indent=2))
    print(f"{sel}: added {len(new_entries)} unpadded aliases")
