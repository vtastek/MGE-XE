# velchurn — the slot-churn fixture

An MWSE mod that **drives the player** so the minimized perf harness can exercise code paths a
pinned static save never reaches: cell transitions, and parts leaving and re-entering the drawn set.

Written for MB-1b (skinned object velocity), whose bone-palette **pairing key** exists to survive
slot recycling. Recycling is a rare *event*, the heartbeat samples one frame in 300, and the harness
loads a fixed save and cannot walk — so the key's rejection counters read zero whether the key worked
or was dead code. This makes them read something.

## Deploy

```
cp -r mgeHost64/velchurn "/mnt/c/mgem/morrowind64/Data Files/MWSE/mods/mgexe/"
```

## ⚠ Arming — it is INERT without the marker

The mod returns immediately, registering nothing, unless a file named `ACTIVE` sits beside
`main.lua`. Running it unarmed during real play would be indistinguishable from possession.

```bash
touch "/mnt/c/mgem/morrowind64/Data Files/MWSE/mods/mgexe/velchurn/ACTIVE"   # arm
bash mgeHost64/forge-perf-run.sh 8 300 <save>.ess "" "mvEnable=1"
rm -f "/mnt/c/mgem/morrowind64/Data Files/MWSE/mods/mgexe/velchurn/ACTIVE"   # DISARM — always
```

Disarm even if the run crashed. Check `MWSE.log` for `[velchurn] ARMED` to confirm which state a
session ran in; the host log cannot tell you.

## What it does

A 0.4 s real timer, 24-step cycle: rotate 10°/tick ×12, walk forward ×4, back ×4, jump, cross a load
door. It never acts in `menuMode()` (a teleport issued mid-load is how a fixture corrupts a save) and
takes each door's *own* destination marker, so the player lands where MW would have put them.

## ⚠ The fixture must not manufacture the signal the instrument detects

First version stepped the player **128 units per tick** by writing `position` directly. That is a
teleport, not a walk — ~160 m/s — and the player's own skinned body legitimately moves that far, so
`maxBoneDelta` peaked at **exactly 128.09 u with the pairing key intact**. The reading was correct;
the *motion* was fake, and it sat right in the band the instrument uses to separate animation-scale
from cell-scale. Now 16 u/tick (~20 m/s, a brisk run) and 10° instead of 30°.

A violent yaw has the same problem one layer up: 30°/frame puts a perfectly correct ~1134 px vector
into `mv final: max`, under which a real tear hides completely.

## Result it produced (2026-09-03, MB-1b)

Same save, same fixture, `objVelSkinIgnoreGen` the only difference:

| arm | `stale` rejects | worst accepted bone delta |
|---|---|---|
| key intact | 18–93 | **185 u** @ \|dBake\|=0 |
| generation clause disabled | 0 (all accepted) | **9307 u** @ \|dBake\|=0.23 |

A 50x separation with the origin motionless in both arms, so the rebase is not involved: the
generation clause is what stops a bone claiming to have crossed the cell in one frame.
