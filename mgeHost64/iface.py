"""
Interface of a candidate region of renderScene: which names it needs from outside, and which
of its own names are read after it.

Fixes the two blind spots that each cost a build during batch 1:
  - renderScene's own PARAMETERS are declared in the signature, not the body (missed waterParams)
  - a multi-declarator line declares more than one name (invented eyeAbsZ, whose real declaration
    was the third name on `const float eyeAbsX = ..., eyeAbsY = ..., eyeAbsZ = ...`)
Scope is still ignored, so treat the answer as a shortlist the compiler then confirms.
"""
import re, sys

PATH = '/mnt/c/projects/mgexe/MGE-XE/mgeHost64/forgerender.cpp'
raw = open(PATH, encoding='utf-8').read().split('\n')

i = [k for k, l in enumerate(raw) if l.startswith('    bool renderScene(const float* viewProj')][0]
# signature runs until the line ending in '{'
sigend = i
while not raw[sigend].rstrip().endswith('{'):
    sigend += 1
SIG = '\n'.join(raw[i:sigend + 1])
PARAMS = set(re.findall(r'\b([A-Za-z_]\w*)\s*(?:,|\)|\s*$)', SIG.split('(', 1)[1]))
PARAMS = set(re.findall(r'(?:const\s+)?[A-Za-z_][\w:]*\s*\**\s*\**\s*([A-Za-z_]\w*)\s*(?=[,)])',
                        SIG.split('(', 1)[1]))

d = 0; st = False; end = None
for k in range(i, len(raw)):
    d += raw[k].count('{') - raw[k].count('}')
    if not st and '{' in raw[k]: st = True
    if st and d <= 0: end = k; break
LO, HI = i + 1, end + 1

def strip(lines):
    out = []; blk = False
    for l in lines:
        r = []; j = 0; s = None
        while j < len(l):
            c = l[j]; n = l[j+1] if j+1 < len(l) else ''
            if blk:
                if c == '*' and n == '/': blk = False; j += 2; continue
                j += 1; continue
            if s:
                if c == '\\': j += 2; continue
                if c == s: s = None
                j += 1; continue
            if c == '/' and n == '/': break
            if c == '/' and n == '*': blk = True; j += 2; continue
            if c in '"\'': s = c; j += 1; continue
            r.append(c); j += 1
        out.append(''.join(r))
    return out

body = strip(raw[LO-1:HI])
KW = {'if','for','while','switch','return','else','do','case','break','continue','sizeof','new',
      'delete','using','namespace','struct','class','enum','template','typedef','operator',
      'decltype','inline','goto','throw','catch','try','const','static','constexpr','auto',
      'unsigned','signed','volatile','true','false','nullptr','this'}
HEAD = re.compile(r'^\s*(?:const\s+|static\s+|constexpr\s+|volatile\s+|auto\s+|unsigned\s+|signed\s+)*'
                  r'([A-Za-z_][\w:]*\s*(?:<[^;{]*>)?)\s+(.*)$')
NAME = re.compile(r'\**\s*&?\s*([A-Za-z_]\w*)\s*(?:=|\[|\{|,|;|\))')

decl = {}
for k, l in enumerate(body):
    s = l.strip()
    inner = None
    if s.startswith('for') and '(' in s:
        inner = s[s.index('(')+1:]
    m = HEAD.match(inner if inner else l)
    if not m: continue
    typ, rest = m.group(1), m.group(2)
    if typ in KW or typ.split('<')[0] in KW: continue
    # split the declarator list on top-level commas
    depth = 0; cur = ''; parts = []
    for ch in rest:
        if ch in '([{<': depth += 1
        elif ch in ')]}>': depth -= 1
        if ch == ',' and depth == 0: parts.append(cur); cur = ''
        else: cur += ch
    parts.append(cur)
    for p in parts:
        nm = NAME.match(p.strip()) or re.match(r'\**\s*&?\s*([A-Za-z_]\w*)\s*$', p.strip())
        if nm and nm.group(1) not in KW:
            decl.setdefault(nm.group(1), []).append(k)

IDENT = re.compile(r'\b([A-Za-z_]\w*)\b')
MEMBER = re.compile(r'(?:\.|->)\s*([A-Za-z_]\w*)')
def names(k):
    l = body[k]
    toks = set(IDENT.findall(l))
    return toks - set(MEMBER.findall(l))      # drop struct-member reads (the `hi` / `at` trap)

first, last = int(sys.argv[1]), int(sys.argv[2])
a, b = first - LO, last - LO + 1
inside = set();  after = set()
for k in range(a, b): inside |= names(k)
for k in range(b, len(body)): after |= names(k)

ins = sorted(n for n, ds in decl.items()
             if any(x < a for x in ds) and n in inside and not any(a <= x < b for x in ds))
pin = sorted(n for n in PARAMS if n in inside)
outs = sorted(n for n, ds in decl.items()
              if any(a <= x < b for x in ds) and n in after and not any(x >= b for x in ds))

print(f"region {first}..{last}  ({last-first+1} lines)")
print(f"\n  renderScene PARAMS used : {', '.join(pin)}")
print(f"\n  locals IN  ({len(ins)}):")
for n in ins:
    k = [x for x in decl[n] if x < a][-1]
    print(f"      {n:22} {raw[LO+k-1].strip()[:82]}")
print(f"\n  locals OUT ({len(outs)}):")
for n in outs:
    k = [x for x in decl[n] if a <= x < b][0]
    print(f"      {n:22} {raw[LO+k-1].strip()[:82]}")
