# Variable Target Count — One Policy for 1–4 Points (2026-08-30, DD-17)

Completes the multi-target arc: `aegis_split*` handled exactly 3 points;
`aegis_vark` handles **any count 1–4**, K sampled per world. Fine-tuned from
aegis_splitspawn with per-world K~U[1,4] (env feature: target_count_range).

## ONE generalist, evaluated at each fixed target count
| Points | Overall win | Direct raids | Saturation raids | Attrition |
|---|---|---|---|---|
| **1** | **100%** | 100% | 100% | 3% |
| **2** | 98% | 97% | 100% | 11% |
| **3** | 96% | 92% | 100% | 11% |
| **4** | 95% | 91% | 100% | 11% |

## Generality is ~free
- **K=1: 100%** — identical to the dedicated single-target champions. Adding the
  ability to defend 2/3/4 points cost *nothing* on the single-point case.
- **K=3: 96% (92% direct)** — matches the dedicated `aegis_splitspawn` (90–95%).
- Saturation raids **100% at every K**; direct raids degrade gracefully 100→91%
  as the swarm must cover more spread points at once. First intercept 1.4 s.

## Significance
A single deployable policy now covers the whole protected-point range — no
per-scenario model selection. This is the same lesson as the size-invariant
swarm: the ego-centric obs + allocator + emergent coordination generalize over
the *number of targets* just as they do over the *number of drones*. `aegis_vark`
is the multi-target champion; big24/oomval remain the reference for pure
single-target scale (1v1→50v40).

## Artifacts
Champion `aegis_vark` (1–4 targets); replay `aegis_vark_4targets_16x16.gif`
(four gold stars, 16 defenders covering all). 256 worlds, no OOM.
