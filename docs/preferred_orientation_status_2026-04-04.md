# Preferred Orientation Status 2026-04-04

This note records the current state of the exact preferred-orientation (PO) pilot, the WiSE-FT interpolation test, and the next mixed-data experiment.

## Exact PO implementation

- Preferred orientation is now implemented as an exact March-Dollase reweighting inside the Fortran wrapper used by the current `CFML_api` powder-simulation path.
- This is not a profile-space surrogate.
- A direct A/B test with only `po_r` changed produced materially different patterns.
- A reflection-family sanity check with preferred direction `(001)` confirmed that the expected reflection families are enhanced/suppressed consistently with the MD formula.

## PO dataset

- Generated dataset:
  - `/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_200k_po_v1_trainready.hdf5`
- Archive copy:
  - `/corral/nist/PHY26002/williamratcliff/ai_diffraction_archive/trainready/rruff_conditioned_200k_po_v1_trainready.hdf5`
- Effective split sizes:
  - train `162500`
  - val `18750`
  - test `18750`

This pilot only changed preferred orientation.
It did not incorporate a broader SimXRD-style rewrite of background/noise/zero-shift assumptions.

Preferred orientation was added as an extra nuisance axis, not as a replacement for the ordinary Stage-2 dirtiness:

- impurity generation/mixing still runs
- empirical backgrounds still run
- existing noise/line-shape machinery still runs

So the PO corpus remains a dirty geological-style synthetic distribution rather than a clean single-phase texture-only corpus.

## PO training results

All runs started from `ic6gfmvm`.

### 1-epoch PO checkpoint (`cscjfdwk`)

Synthetic held-out:
- decoded Top-1 `7.13%`
- decoded Top-3 `14.23%`
- decoded Top-5 `20.90%`
- aux EG Top-1 `26.40%`
- aux EG Top-3 `52.28%`
- aux EG Top-5 `66.27%`

Real `RRUFF-325`, calibrated aux (`T=5`):
- Top-1 `13.54%`
- Top-5 `40.92%`
- `ECE 0.0131`
- `NLL 3.669`
- `Brier 0.953`

Real `RRUFF-473`, calibrated aux (`T=5`):
- Top-1 `13.53%`
- Top-5 `49.89%`
- `ECE 0.0077`
- `NLL 3.433`
- `Brier 0.940`

### 2 effective epochs via continuation (`dsi7ehiv`)

Synthetic held-out:
- decoded Top-1 `7.39%`
- decoded Top-3 `14.75%`
- decoded Top-5 `22.40%`
- aux EG Top-1 `27.24%`
- aux EG Top-3 `53.75%`
- aux EG Top-5 `68.01%`

Real `RRUFF-325`, calibrated aux (`T=5`):
- Top-1 `12.62%`
- Top-5 `41.23%`
- `ECE 0.0184`
- `NLL 3.671`
- `Brier 0.953`

Real `RRUFF-473`, calibrated aux (`T=5`):
- Top-1 `12.47%`
- Top-5 `49.68%`
- `ECE 0.0134`
- `NLL 3.435`
- `Brier 0.940`

### Current read

- PO helps early.
- The best real-data Top-1 result is the 1-epoch PO checkpoint, not the 2-epoch continuation.
- A second epoch improves the synthetic held-out split slightly, but does not improve real-data calibrated Top-1.

## Split-head validity

The split head remains weak under the current PO pilot.

1-epoch PO:
- `RRUFF-325`: valid `0.92%`, ambiguous `1.23%`, invalid `97.85%`
- `RRUFF-473`: valid `0.85%`, ambiguous `0.85%`, invalid `98.31%`

2-epoch PO:
- `RRUFF-325`: valid `1.85%`, ambiguous `0.92%`, invalid `97.23%`
- `RRUFF-473`: valid `1.48%`, ambiguous `0.63%`, invalid `97.89%`

So exact symbolic decoding is still not the main benefit of PO at this stage.

## WiSE-FT result

We interpolated weights between:
- `9rwv1qly` (broad Stage-2c checkpoint)
- `cscjfdwk` (1-epoch PO checkpoint)

`RRUFF-325` calibrated aux results:
- `alpha=0.3`, Stage-2 prior:
  - Top-1 `11.69%`
  - Top-5 `42.15%`
- `alpha=0.3`, PO prior:
  - Top-1 `11.38%`
  - Top-5 `42.46%`
- `alpha=0.5`, Stage-2 prior:
  - Top-1 `12.62%`
  - Top-5 `41.85%`
- `alpha=0.5`, PO prior:
  - Top-1 `12.62%`
  - Top-5 `41.54%`

This did not recover the desired hybrid.
It improved Top-1 relative to old Stage-2c, but did not retain the 1-epoch PO Top-1 and did not preserve the old Stage-2c Top-5.

## Topology / DAG result

We ran the `RRUFF-325` topological error analysis on the 1-epoch PO checkpoint.

Baseline old Stage-2c (`9rwv1qly_aux_bayes_t5`):
- correct `31`
- wrong `294`
- mean wrong distance `2.72`
- `% wrong within graph distance <= 2`: `38.4%`
- directionality:
  - descendant `191`
  - ancestor `27`
  - branch jump `76`

1-epoch PO (`cscjfdwk_aux_bayes_t5`):
- correct `44`
- wrong `281`
- mean wrong distance `2.51`
- `% wrong within graph distance <= 2`: `53.0%`
- directionality:
  - descendant `157`
  - ancestor `28`
  - branch jump `96`

Interpretation:
- PO improves local/topological closeness of errors.
- PO reduces descendant errors.
- PO does **not** materially increase ancestor errors.
- The stronger “PO rebalances descendant vs ancestor flow” claim is not currently supported.

## Mixed 200k pilot

We also ran a small mixed Stage-2 experiment:

- mixed trainready:
  - `/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_mixed_200k_v1_trainready.hdf5`
- checkpoint:
  - `eeru8svx`

Real `RRUFF-325`, calibrated aux (`T=5`):
- Top-1 `13.23%`
- Top-5 `40.92%`
- `ECE 0.0131`
- `NLL 3.670`
- `Brier 0.953`

Real `RRUFF-473`, calibrated aux (`T=5`):
- Top-1 `13.53%`
- Top-5 `49.05%`
- `ECE 0.0095`
- `NLL 3.438`
- `Brier 0.940`

Split-head validity:
- `RRUFF-325`: valid `1.54%`, ambiguous `0.0%`, invalid `98.46%`
- `RRUFF-473`: valid `1.27%`, ambiguous `0.0%`, invalid `98.73%`

Interpretation:
- the small mixed pilot did not beat the 1-epoch PO checkpoint on `RRUFF-325` Top-1
- it was effectively tied with PO on `RRUFF-473` Top-1
- it did not recover the older Stage-2c `RRUFF-325` Top-5 breadth

So the mixed `200k` result is best read as a compromise/stabilization point, not a clear improvement over the 1-epoch PO model.

## Mixed 200k topology

We ran the same `RRUFF-325` topology analysis on `eeru8svx`.

Mixed `200k` (`eeru8svx_aux_bayes_t5`):
- correct `43`
- wrong `282`
- mean wrong distance `2.59`
- `% wrong within graph distance <= 2`: `49.65%`
- directionality:
  - descendant `153`
  - ancestor `30`
  - branch jump `99`

Relative to the earlier checkpoints:
- Stage-2c: descendant `191`, ancestor `27`, branch jump `76`
- PO 1 epoch: descendant `157`, ancestor `28`, branch jump `96`
- Mixed `200k`: descendant `153`, ancestor `30`, branch jump `99`

Interpretation:
- mixed training keeps the main PO benefit of reducing descendant errors and tightening locality
- it produces the lowest descendant count of the three models
- but it does not reduce branch jumps relative to PO

So the small mixed pilot still does not solve the branch-jump issue.

## Current working hypothesis

The best interpretation so far is:
- exact PO acts as a useful realism regularizer,
- improves early calibrated Top-1 on real data,
- and makes errors more topologically local,
- but a pure all-PO Stage-2 continuation is too narrow to preserve the broader Stage-2c Top-5 behavior.

The stronger current topology interpretation is that many of the remaining branch jumps are likely local “texture aliasing” or cousin-confusion effects in 1D powder space rather than simple wild failures:

- major reflection families can be heavily suppressed by PO
- once those reflections are weakened, the remaining 1D barcode can resemble a nearby crystallographic cousin
- this is consistent with the strong `%<=2` locality gains despite elevated branch-jump counts

So the important result is not that branch jumps vanished, but that the errors stayed physically local while descendant bias weakened.

## Data-order / shuffle note

The mixed HDF5 builders write standard and PO samples contiguously within each split.

The active ViT training pipeline still shuffles training examples at runtime:
- single-process runs use `DataLoader(..., shuffle=True)` when no sampler is present
- distributed runs reset a shuffled sampler each epoch

## Final large mixed run

We then trained the final large mixed model from the same uniform base checkpoint `ic6gfmvm` using a dual-source loader rather than physically merging the source HDF5 files:

- standard source: `rruff_conditioned_2346k_v1_trainready.hdf5`
- PO source: `rruff_conditioned_500k_po_v1_trainready.hdf5`
- effective training mixture: approximately `2.346M` standard + `500k` PO
- checkpoint/run: `82ept35h`
- training length: exactly `1` epoch

Real `RRUFF-325`, calibrated Bayesian aux (`T=5`):
- Top-1 `10.46%`
- Top-5 `43.69%`
- `ECE 0.0381`
- `NLL 3.688`
- `Brier 0.957`

Real `RRUFF-473`, calibrated Bayesian aux (`T=5`):
- Top-1 `9.94%`
- Top-5 `50.74%`
- `ECE 0.0429`
- `NLL 3.449`
- `Brier 0.943`

Split-head validity recovered sharply relative to the pure-PO runs:

- `RRUFF-325`: valid `47.38%`, ambiguous `29.85%`, invalid `22.77%`
- `RRUFF-473`: valid `49.26%`, ambiguous `30.87%`, invalid `19.87%`

This is the strongest current balanced model:

- it recovers broad `Top-5` coverage (`43.69%` on `RRUFF-325`, slightly above the old Stage-2c `43.08%`)
- it still improves Top-1 over the old non-PO Stage-2c model (`10.46%` vs `9.54%`)
- it restores split-head self-consistency dramatically compared with the pure-PO runs

## Final large mixed topology

Large mixed (`82ept35h_aux_bayes_t5`) on `RRUFF-325`:
- correct `34`
- wrong `291`
- mean wrong distance `2.46`
- `% wrong within graph distance <= 2`: `51.55%`
- directionality:
  - descendant `165`
  - ancestor `31`
  - branch jump `95`

Comparison across the main models:

- Stage-2c:
  - Top-1 / Top-5 `9.54 / 43.08`
  - split valid `~0.6%`
  - descendant / ancestor / branch `191 / 27 / 76`
  - `%<=2` `38.4%`
  - mean distance `2.72`
- PO-only, 1 epoch:
  - Top-1 / Top-5 `13.54 / 40.92`
  - split valid `0.92%`
  - descendant / ancestor / branch `157 / 28 / 96`
  - `%<=2` `53.0%`
  - mean distance `2.51`
- Mixed `200k`:
  - Top-1 / Top-5 `13.23 / 40.92`
  - split valid `1.54%`
  - descendant / ancestor / branch `153 / 30 / 99`
  - `%<=2` `49.65%`
  - mean distance `2.59`
- Final large mixed:
  - Top-1 / Top-5 `10.46 / 43.69`
  - split valid `47.38%`
  - descendant / ancestor / branch `165 / 31 / 95`
  - `%<=2` `51.55%`
  - mean distance `2.46`

Interpretation:

- the final large mixed run is the best balanced model
- it preserves the strong locality gain seen in the PO-informed runs
- it reduces descendant bias relative to Stage-2c (`191 -> 165`)
- it modestly increases ancestor predictions (`27 -> 31`)
- branch jumps remain elevated, but within a much tighter local regime than the non-PO baseline

The best current interpretation is still local “texture aliasing” or cousin-confusion in 1D powder space:

- severe preferred orientation suppresses whole reflection families
- once those families are weakened, the remaining 1D trace can resemble a nearby crystallographic cousin
- the network therefore makes local lateral hops rather than globally implausible jumps

## Manuscript-ready LaTeX

```latex
\begin{table}[t]
\centering
\caption{Real-data performance and topological behavior on RRUFF-325 under calibrated Bayesian auxiliary decoding.}
\label{tab:po_topology_rruff325}
\begin{tabular}{lccccc}
\toprule
Model curriculum & Top-1 / Top-5 & Split valid & Desc./Anc./Branch & $\leq$2 hops & Mean DAG dist. \\
\midrule
Stage-2c (no PO)      & 9.54 / 43.08  & $\sim$0.6\% & 191 / 27 / 76 & 38.4\%  & 2.72 \\
PO-only, 1 epoch      & \textbf{13.54} / 40.92 & 0.92\% & 157 / 28 / 96 & \textbf{53.0\%} & 2.51 \\
Mixed 200k pilot      & 13.23 / 40.92 & 1.54\% & 153 / 30 / 99 & 49.65\% & 2.59 \\
Final large mixed run & 10.46 / \textbf{43.69} & \textbf{47.38\%} & 165 / 31 / 95 & 51.55\% & \textbf{2.46} \\
\bottomrule
\end{tabular}
\end{table}
```

```latex
\paragraph{Preferred orientation and topological error structure.}
To test whether subtractive experimental noise changes the model's topological behavior, we introduced exact March--Dollase preferred orientation (PO) during Stage-2 realism adaptation while leaving Stage-1 uniform pretraining unchanged. A pure-PO fine-tuning ablation produced the best calibrated RRUFF-325 Top-1 accuracy (13.54\%), confirming that explicit modeling of texture-suppressed reflections improves exact ranking on real diffraction patterns. However, this came at a cost: Top-5 breadth fell relative to the non-PO Stage-2c model (40.92\% vs.\ 43.08\%), split-head symbolic validity collapsed below 1\%, and lateral branch-jump errors remained elevated.

We therefore trained a larger mixed curriculum combining the full standard RRUFF-conditioned corpus with a 500k PO corpus. This large mixed model is the best overall deployment model. On RRUFF-325 it achieved calibrated Top-1 / Top-5 = 10.46\% / 43.69\%, slightly exceeding the earlier non-PO Stage-2c Top-5 while still improving Top-1 over the non-PO baseline. On RRUFF-473 it reached 9.94\% / 50.74\%. Most strikingly, strict split-head validity recovered from 0.92\% in the PO-only run to 47.38\% on RRUFF-325 and 49.26\% on RRUFF-473, indicating that the mixed curriculum restores internal symbolic consistency while retaining the benefits of explicit PO exposure.

Topological DAG analysis clarifies the tradeoff. Relative to Stage-2c, the final mixed model substantially reduced the descendant-heavy bias (191 $\rightarrow$ 165 descendant errors), modestly increased ancestor predictions (27 $\rightarrow$ 31), and tightened the local error neighborhood: 51.55\% of wrong predictions fell within $\leq$2 hops of the truth, versus 38.4\% for Stage-2c, and the mean wrong-path distance dropped from 2.72 to 2.46, the best observed value. Branch jumps remained common (95), but they occurred within a much tighter local regime than in the non-PO baseline. This pattern is consistent with a \emph{texture aliasing} interpretation: when PO suppresses entire reflection families, a 1D powder trace may become locally ambiguous between nearby crystallographic cousins, producing lateral hops without causing globally implausible predictions.

Taken together, these results support a two-part conclusion. First, explicit preferred-orientation physics matters: the PO-only ablation demonstrates that missing-peak realism improves exact ranking and reduces conservative descendant errors. Second, PO should not dominate the realism curriculum. The best final model is the large mixed run, where textured patterns are present often enough to regularize missing-peak reasoning, but not so often that the model loses nuisance breadth or symbolic consistency. In practice, the mixed curriculum yields the strongest balance among calibrated Top-5 coverage, improved Top-1, restored split-head validity, and the tightest topological error radius.
```
- distributed runs use `DistributedSampler(..., shuffle=True)` and reset the sampler each epoch via `set_epoch(...)`

So the model does not consume the epoch in a fixed “all standard then all PO” order.

## Large follow-on in progress

The expensive follow-on PO generation is complete:

- new `300k` PO trainready:
  - `/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_300k_po_v1_trainready.hdf5`
- merged `500k` PO trainready:
  - `/scratch/09870/williamratcliff/ai_diffraction_generated/rruff_conditioned_500k_po_v1_trainready.hdf5`

The final large mixed build now targets:
- standard Stage-2 corpus: `2.346M`
- PO corpus: `500k`
- final mixed trainready scale: `2.5M`

This is the decisive next experiment because the small mixed pilot strongly suggests that scale matters:
- PO helps Top-1 and local topology
- but broader nuisance entropy is still needed to recover Top-5 breadth

Planned training/eval recipe:
- start from `ic6gfmvm`
- train `1` epoch at learning rate `1e-5`
- sweep calibration temperature (`T=2,3,4,5,6`) after training before declaring the final calibrated `Top-1/Top-5`
