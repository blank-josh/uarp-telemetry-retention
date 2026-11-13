# UARP: Utility-Aware Retention Policies for bounded telemetry buffers

Evaluates retention policies (FIFO, LIFO, RandomEviction, Reservoir, TimeStratified, A-Res weighted reservoir,
ValueBiased, UtilityAware, StratifiedUtility, UtilityAware-Tuned) on real labelled time-series data. For every policy
the retained buffer is used to train downstream detectors (Isolation Forest, Random Forest, robust-z), which are
scored on a held-out later period.

## Where to put the dataset

```
uarp_project/
  data/
    skab/            <- unzip the Kaggle SKAB download HERE (any sub-folders are fine, CSVs are found recursively)
    pump/sensor.csv  <- optional second dataset: Kaggle "Pump Sensor Data", file name must be sensor.csv
```
Change paths, column names and label values in `config.py` (`DATASET_CONFIGS`). Nothing else needs editing.
Pick which datasets run with `DATASETS = ["skab"]` (add `"pump"` later), or `--dataset skab pump`.

## Run

```
pip install -r requirements.txt      # Colab already has everything
python run_all.py --fast             # 1-minute smoke test
python run_all.py                    # full run: 10 seeds, 4 buffer sizes, ablations, tuning, timing
```

## Protocol (no leakage)
* Data is one chronological stream. Splits: first 60% = stream S (policies see it once, in order), next 15% =
  validation V (weight tuning only), last 25% = test T (final numbers only). SKAB is split on file boundaries.
* Policies are label-free: they only see the sample position and the scorer components, never the label.
* Scalers and detectors are fit on the retained buffer only. T is never seen by a policy or a scaler.
* Buffer size K = ratio x |S| (0.5%, 1%, 2%, 5%); headline = 2%. Seed 0 = untrimmed stream; seeds 1..9 trim 0-5%
  from the start of S and reseed the stochastic policies.
* Design choice: a sample's anomaly / deviation / rarity components are computed once, at arrival, and frozen;
  only the recency term decays with age. This makes policies O(K) per arrival and lets the same scorer output be
  reused across policies. State this in the paper.

## Outputs (`results/<dataset>/`)
`metrics_main.csv`, `metrics_by_K.csv`, `ablation.csv`, `weight_tuning.csv`, `significance.csv`, `systems.csv`,
`runs_all.csv` (every individual run), `results.json`, `run_log.txt`, `RESULTS_SUMMARY.txt`,
`table_main.tex`, `table_ablation.tex`, and `fig_*.png` (300 dpi).

## Known limitations to discuss in the paper
* SKAB has a high anomaly rate (roughly a fifth to a third of rows), so a buffer that favours anomalies can
  saturate with them. This hurts unsupervised detectors trained on the buffer. Report it.
* The robust-z detector is closely related to the policy's anomaly term; treat its F1 as secondary and lead
  with Isolation Forest / Random Forest.
* The supervised Random Forest uses labels of retained samples (retrospective labelling); the policy itself never does.
* Pump Sensor has very few failure events, so event-level statistics there are weak.
