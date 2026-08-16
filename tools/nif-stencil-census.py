#!/usr/bin/env python3
"""Census of REAL NiStencilProperty use across a Morrowind install.

Every MW shape can carry a NiStencilProperty, but the overwhelming majority use it only for
`draw_mode` (DRAW_BOTH = two-sided) with the stencil test itself DISABLED — that is the flag the
geometry cache already reads. This census finds the shapes that actually drive the STENCIL BUFFER:
stencil_enabled != 0 with a test/op combination that does something (the "fake hole" portal trick in
comGravePit.nif being the motivating case).

Usage:  python3 tools/nif-stencil-census.py "/mnt/c/mgem/morrowind64/Data Files/meshes"
"""
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "io_scene_mw-master", "lib"))
from es3.nif import NiStream  # noqa: E402


def walk(node, out, depth=0):
    for pr in getattr(node, "properties", []) or []:
        if type(pr).__name__ == "NiStencilProperty" and pr.stencil_enabled:
            out.append((getattr(node, "name", ""), pr))
    for c in getattr(node, "children", []) or []:
        if c is not None:
            walk(c, out, depth + 1)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/mgem/morrowind64/Data Files/meshes"
    files = []
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith(".nif"):
                files.append(os.path.join(dirpath, n))
    files.sort()
    print(f"scanning {len(files)} NIFs under {root}", flush=True)

    combos = Counter()
    per_combo_files = defaultdict(set)
    nEnabled = 0
    nFilesEnabled = 0
    nFail = 0

    for i, f in enumerate(files):
        if i and i % 5000 == 0:
            print(f"  ...{i}/{len(files)}", flush=True)
        try:
            s = NiStream()
            s.load(f)
        except Exception:
            nFail += 1
            continue
        hits = []
        try:
            walk(s.root, hits)
        except Exception:
            nFail += 1
            continue
        if not hits:
            continue
        nFilesEnabled += 1
        for shapeName, pr in hits:
            nEnabled += 1
            key = (pr.stencil_function.name, pr.stencil_ref & 0xFF,
                   pr.fail_action.name, pr.pass_z_fail_action.name, pr.pass_action.name)
            combos[key] += 1
            per_combo_files[key].add(os.path.relpath(f, root))

    print()
    print(f"NIFs parsed        : {len(files) - nFail}  (parse failures {nFail})")
    print(f"NIFs with enabled  : {nFilesEnabled}")
    print(f"shapes with enabled: {nEnabled}")
    print()
    print("func / ref&0xFF / fail / zfail / pass                      shapes  files")
    for key, n in combos.most_common():
        fn, ref, fa, za, pa = key
        files_ = per_combo_files[key]
        print(f"  {fn:<18} ref={ref:<4} {fa:<16} {za:<16} {pa:<16} {n:>6}  {len(files_)}")
    print()
    print("--- files per NON-TRIVIAL combo (anything that is not ALWAYS/KEEP/KEEP/KEEP) ---")
    for key, n in combos.most_common():
        fn, ref, fa, za, pa = key
        if fn == "TEST_ALWAYS" and fa == za == pa == "ACTION_KEEP":
            continue
        print(f"[{fn} ref={ref} {fa}/{za}/{pa}]  {n} shapes")
        for p in sorted(per_combo_files[key])[:40]:
            print(f"    {p}")
        if len(per_combo_files[key]) > 40:
            print(f"    ... +{len(per_combo_files[key]) - 40} more files")


if __name__ == "__main__":
    main()
