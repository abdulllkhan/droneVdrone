# OOM Fix — Validated at Full 512 Worlds (2026-08-29, DD-17)

## The fix (commit 29a874d)
The recurring `exit 137` (4 kills across nights) was NOT RAM exhaustion — steady
RSS was always low. Root cause: `step_wait` allocated a fresh `wp.array()` for
drone velocities EVERY step, and `_upload_resets` reallocated init/mask/mass
arrays on EVERY world reset. At high world counts that churn spiked the unified
(VRAM+RAM) pool into transient OOMs. Fix: preallocate those GPU buffers once,
copy in-place via `.assign()`. Bit-identical (21/21 tests).

## Validation: the config that killed us 4× now runs clean
512 worlds × n_max=20 (10,240 agent streams), 30M-step stage-3 fine-tune:
- **exit 0, NO OOM**, RAM flat ~12 GB throughout (prior runs died ~20M steps in)
- 171k agent-steps/s in a microbench; ~1.1 GB peak host RSS
- Gate: **win 0.97** (256 eps), 7.7 intercepts/episode

## Bonus champion `aegis_oomval` (full-worlds terrain generalist)
| Eval | Win | Notes |
|---|---|---|
| Maze 10v10 + 20v20 | **1.00** | matches largeP; terrain deaths 20% (unshaped this run) |
| Scale 40v30 + 50v40 | **1.00** | 90-drone battles, 6% losses |

## Impact: the ≤256-world cap is LIFTED
Every large run since night 1 was throttled to 256/384 worlds to dodge the OOM.
Full 512 worlds (≈2× parallelism, ~2× faster large-N training) is available
again for all future work. This unblocks the bigger ambitions (30v30+, deeper
population loops, larger adversarial pools) at full throughput.
