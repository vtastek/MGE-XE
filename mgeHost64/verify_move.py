"""
Phase 3 oracle (tasks/forge-host-decomposition.md): prove each promoted pass is a VERBATIM move.

A build proves the code compiles. This proves it is THE SAME CODE: each promoted function's body is
diffed line for line against the region it came from, as that region stands in a baseline commit.
The invariant it enforces is the one that matters — no gpuPhase call moves relative to the work it
brackets — so a promotion cannot change what a phase measures.

⚠ RUN THIS BEFORE COMMITTING, NOT AFTER. The default baseline is HEAD, which holds the passes in
their ORIGINAL form only until the batch is committed. Once committed, HEAD has the promoted form
and a pass can no longer be checked against it. To re-check an older batch, pass that batch's
baseline commit:

    python3 mgeHost64/verify_move.py            # current working tree vs HEAD
    python3 mgeHost64/verify_move.py a8cc463a   # batch 1's baseline (commit 39a73d84 promoted it)

CASES holds the batch currently in the working tree. Landed batches are listed below it, with the
baseline they were verified against, so the record survives even though the check cannot be re-run
from HEAD.
"""
import subprocess, sys, difflib

PATH = 'mgeHost64/forgerender.cpp'
REPO = '/mnt/c/projects/mgexe/MGE-XE'
BASE = sys.argv[1] if len(sys.argv) > 1 else 'HEAD'

# name, baseline first line, baseline last line, current signature prefix, banner kept at call site,
# extra lines past the last anchor
CASES = [
    ("Frame epilogue",
     "        // --- M1: SNAPSHOT THIS FRAME'S CAMERA FOR NEXT FRAME'S REPROJECTION -----------------------",
     '            g_aplLandN = g_aplWaterN = 0;',
     '    void frameEpilogue(', False, 1),
]

# Landed and verified, no longer checkable from HEAD (each was VERBATIM against the baseline named):
#   baseline a8cc463a -> commit 39a73d84 (batch 1)
#       passDistantLightGlow           18 lines
#       passVolumetricFogAndBackstop   52 lines
#       passResolvePreFilter           41 lines
#       passForgeWaterSurface         195 lines
#   baseline 39a73d84 -> commit 1c10da77 (batch 2)
#       passLinearizeAndGtao          370 lines  (+ the single-output `return aoBlockRan;`)
#       passPointLightShadowFaces     600 lines
#       passZPrepass                  419 lines
#       passHiZPrologue                95 lines

base = subprocess.run(['git', '-C', REPO, 'show', f'{BASE}:{PATH}'],
                      capture_output=True, text=True, check=True).stdout.split('\n')
cur = open(f'{REPO}/{PATH}', encoding='utf-8').read().split('\n')


def region(lines, first, last, extra=0):
    a = [i for i, l in enumerate(lines) if l == first]
    b = [i for i, l in enumerate(lines) if l == last]
    assert len(a) == 1, f"first line: {len(a)} matches: {first!r}"
    assert len(b) == 1, f"last line: {len(b)} matches: {last!r}"
    return lines[a[0]:b[0] + 1 + extra]


def body_of(lines, sig_prefix):
    i = [k for k, l in enumerate(lines) if l.startswith(sig_prefix)]
    assert len(i) == 1, f"signature {sig_prefix!r}: {len(i)} matches"
    # a signature may wrap over several lines; the body starts after the one ending in '{'
    s = i[0]
    while not lines[s].rstrip().endswith('{'):
        s += 1
    s += 1
    e = s
    while lines[e] != '    }':
        e += 1
    return lines[s:e]


fail = 0
for name, first, last, sig, drop_banner, extra in CASES:
    was = region(base, first, last, extra)
    if drop_banner:
        was = was[1:]                     # the banner stayed at the call site
    now = body_of(cur, sig)
    # a single-output pass appends a `return`, which is stated in its own comment block
    if len(now) > len(was) and any('return ' in l for l in now[len(was):]):
        appended = now[len(was):]
        now = now[:len(was)]
        note = f"  (+{len(appended)} appended lines: the single-output return)"
    else:
        note = ""
    if was == now:
        print(f"  VERBATIM  {len(now):5} lines   {name}{note}")
    else:
        fail += 1
        print(f"  ⚠ DIFFERS            {name}")
        for l in difflib.unified_diff(was, now, BASE, 'now', lineterm='', n=1):
            print('   ', l)

sys.exit(1 if fail else 0)
