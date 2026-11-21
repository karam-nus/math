"""Regenerate every figure in assets/figures/ (light + dark variants).

    pip install -r requirements.txt
    python scripts/figures/build_all.py            # all figures
    python scripts/figures/build_all.py ch04 ch31  # only names containing these
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from style import REGISTRY  # noqa: E402

PARTS = ["part_a", "part_b", "part_c", "part_d", "part_e", "part_f", "part_g", "part_h"]

for mod in PARTS:
    try:
        importlib.import_module(mod)
    except ModuleNotFoundError as e:
        if e.name != mod:
            raise

filters = sys.argv[1:]
names = [n for n in REGISTRY if not filters or any(f in n for f in filters)]
for n in names:
    REGISTRY[n]()
    print("built", n)
print(f"{len(names)} figure(s) x 2 themes")
