"""
Prove each promoted pass is a VERBATIM move: the function's body, line for line, must equal
the region that used to sit in renderScene in git HEAD.

This is the Phase 3 oracle. A build only proves the code compiles; this proves it is the same
code. Anything that differs is printed as a unified diff rather than summarised away.
"""
import subprocess, sys, difflib

PATH = 'mgeHost64/forgerender.cpp'
REPO = '/mnt/c/projects/mgexe/MGE-XE'

head = subprocess.run(['git', '-C', REPO, 'show', f'HEAD:{PATH}'],
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

CASES = [
    # name, HEAD first line, HEAD last line, current signature prefix, banner kept at call site?
    ("PHASE F glow",
     '        // ===================== PHASE F: DISTANT-LIGHT GLOW BILLBOARDS =====================',
     '        gpuPhaseEnd(kGpuPhaseColorGlow);',
     '    void passDistantLightGlow(', True),
    ("VolFog + W8 backstop",
     '        // ===================== VOLUMETRIC height fog / sun shafts =====================',
     '        gpuPhaseEnd(kGpuPhaseVolFog);',
     '    void passVolumetricFogAndBackstop(', True),
    ("Resolve pre-filter",
     "        // ===================== THE RESOLVE'S COMPUTE PRE-FILTER ==================================",
     '            gpuPhaseEnd(kGpuPhaseResolveFilter);',
     '    void passResolvePreFilter(', True),
    ("WT1 Forge water surface",
     '        gpuPhaseBegin(kGpuPhaseWater);',
     '        gpuPhaseEnd(kGpuPhaseWater);',
     '    void passForgeWaterSurface(', False),
]

fail = 0
for name, first, last, sig, drop_banner in CASES:
    was = region(head, first, last, 1 if 'pre-filter' in name else 0)
    if drop_banner:
        was = was[1:]                      # the banner stayed at the call site
    now = body_of(cur, sig)
    if was == now:
        print(f"  VERBATIM  {len(now):5} lines   {name}")
    else:
        fail += 1
        print(f"  ⚠ DIFFERS            {name}")
        for l in difflib.unified_diff(was, now, 'HEAD', 'now', lineterm='', n=1):
            print('   ', l)

# and the call site must be exactly banner + one call
print()
for name, first, last, sig, _ in CASES:
    i = [k for k, l in enumerate(cur) if l == first]
    assert len(i) == 1
    print(f"  call site: {cur[i[0]+1].strip()[:76]}")

sys.exit(1 if fail else 0)
