# CS60055 Challenge 2 - Cross-Room Few-Shot Adaptation from WiFi CSI

Pure-Python pipeline: PCAP/CSV ingestion -> CSI decoding -> QC -> common 256-column CSV -> 150x256 windows ->
normalization -> self-supervised encoder -> room fingerprint -> cross-room few-label adaptation -> figures.

## Setup & run
```bash
pip install -r requirements.txt
# edit configs/experiment.yaml: paths.data_root -> your CSI-HAR-Hackathon folder (read-only, never committed)
python run_pipeline.py --config configs/experiment.yaml --stage all --workers 4     # or one stage at a time:
#   --stage ingest | windows | train | tasks
python run_episode.py --episode_dir EPISODE/ --out predictions.csv [--support_csv support.csv]
pytest -q tests
```
Smoke test without the dataset: `python tools/make_mock_data.py --out /tmp/mock_csi` (synthetic, same layout/format;
numbers on it are meaningless). `python tools/inspect_sources.py <files>` implements Hint 1.

## Layout
| file | role |
|---|---|
| `src/pcap_reader.py` | classic-PCAP + Ethernet/IPv4/UDP extraction (struct only) |
| `src/csi_decode.py` | Nexmon header parse + BCM4366c0 packed-float -> complex CSI (own implementation) |
| `src/amplitude.py`, `src/cleaning.py` | amplitude, row-level QC with per-capture report |
| `src/data_index.py` | capture discovery, Room-A supplied validation, Room-B seed-42 capture-level split |
| `src/ingest.py` | writes `intermediate_csv/room_X/<split>/<capture>.csv` (256 cols) + `manifest.csv`; windows + `index.csv` provenance |
| `src/windowing.py`, `src/normalization.py` | deterministic windows; log1p + per-subcarrier z-score fitted on **dev only** |
| `src/encoder.py` | 1-D CNN + SimCLR (NT-Xent) pretraining; **no labels used** |
| `src/features.py` | hand-crafted baseline, room-fingerprint features |
| `src/adapt.py`, `src/tasks.py` | few-label adaptation, Tasks 1-3, routing study, metrics |
| `src/figures.py`, `src/pipeline.py`, `src/inference.py` | figures, stages, shared inference path |

Outputs in `results/`: `results.csv` (`experiment,target_room,K,metric,value,method`), `val_predictions.csv`
(`window_id,room,label,confidence`), confusion matrices + per-class P/R/F1 at K=5, and the 5 required figures.
Provenance: every window row in `windows/index.csv` stores capture id, start offset, stride and `preproc_hash`
(hash of the decode/qc/split/window/normalization config).

## What we found in the real data 
1. **Room-A PCAPs hold two interleaved frame types.** Frames are 1084 B (18 B Nexmon header + 256 x uint32). Every second
   packet has Nexmon seq `0xFFFF`; its energy is confined to the ~64 central subcarriers (rest ~3-9), while the other
   half shows a smooth 80 MHz response resembling Room B. Captures hold only ~108 full-band packets, so a
   150-observation window is impossible without the narrow-band ones. Default `decode.stream_policy: all` keeps them (in
   temporal order); `full_band_only` drops them (then use a shorter `window.length`).
   **Consequence:** the period-2 pattern is absent in Room B, so room identification is probably near-trivially
   solvable from the CSI itself. Report Task 1 with this caveat (it is a data property, not file metadata).
2. **Decoding check.** Sign-magnitude mantissas with the exponent added gave the smoothest frequency response
   (mean adjacent |dlog amp| 0.26 vs 0.50-0.59 for two's-complement / flipped-exponent variants) and a median/max
   ratio (0.15) close to Room B (0.13). Verified on the 6 provided PCAPs only; please re-check on the full set.
3. **Subcarrier order.** Raw Room-A order already has the DC null at index ~128 like the Room-B CSVs, so `fftshift: false`.
4. **QC.** On the 6 provided PCAPs and 5 CSVs, no non-finite/all-zero/duplicate rows were found. All counts are logged
   per capture in `manifest.csv`.

## Design decisions & assumptions 
* **Room-B validation split** (*assumption*): per class, `sorted(filenames)` then `random.Random(42).sample(names, 25)`.
  The PDF does not fix the exact sampler; if the TAs use another (e.g. numpy), change `data_index.index_room_b`.
* Windows: dev stride 75 (overlap inside a capture only), validation stride 150 (non-overlapping, start at 0). Captures
  with <150 valid observations are skipped and listed in `windows/skipped_captures.csv` (`run_episode.py` pads by default).
* Metrics are window-level. Room-A validation contributes ~1 window/capture, Room-B ~3.
* Task 2 pools both rooms into one 8-label space; linear probe = logistic regression on frozen features. Baselines: random-weight
  encoder (same architecture) and hand-crafted features.
* Task 3: target label set = shared + exclusive. Shared classes are learned from the **source room's dev labels only**; exclusive
  classes from K target **dev captures** (all windows of those captures; K = captures, not windows). Per-room embedding
  standardization uses that room's *unlabeled* dev windows. Mean +/- std over `adapt.n_draws` random support draws
  (`*_std` metrics). Source-exclusive classes are not used (outside the target label set).
* Routing study (pooled A+B validation, 8 labels): `predicted_room_routing` (router picks the room-specific head),
  `oracle_room`, `no_routing` (both adapted heads run on every window, most confident wins).
* Room classifier (Task 1) uses per-subcarrier temporal mean/std (length-robust; trained on crops of 10-150 observations),
  not file extension, folder, length or packet count. The extension is used *only* by `run_episode.py` to pick the parser.
* Labels come from file/folder names and are only supervision targets; validation data is never used to fit anything
  (normalizer, SSL, heads, room stats, router are all dev-only).

## Known limitations
Room vs session confound; activity vs location confound; two environments only; capture-disjoint not subject-disjoint
(see Challenge PDF section 12). Real-dataset results have not been produced here: only a 6-PCAP / 5-CSV subset
was available, so full-scale hyper-parameters (SSL epochs, K draws, window stride) still need tuning on validation.

## AI-use disclosure
Initial code was drafted with Claude (Anthropic). Every member must read, understand and be able to explain each stage;
update this section with your group's actual disclosure and contribution statement as required by the course.
