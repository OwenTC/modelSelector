# Exhaustive search heuristic benchmark

Run the commands from the ModelSelector repository root. Use a Python environment with the project's dependencies installed. The plotting and reporting command also needs Matplotlib and ReportLab.

```bash
python -m pip install -e . matplotlib reportlab
```

## Submit to Slurm

```bash
python -m examples.synthetic_search.benchmark plan runs/synthetic-search
sbatch --array=0-191%32 examples/synthetic_search/benchmark.sbatch runs/synthetic-search
```

Use the array bounds printed by `plan` if you change the settings. The default plan has 192 tasks. Each task runs all 15 methods on one dataset and writes one result file atomically. You can resubmit missing tasks or the whole array. Workers skip completed cases. Set `PYTHON` to the path of a cluster Python environment if it differs from the `python` on your path. Supply any account, partition, and time overrides required by your site to `sbatch`.

After the tasks complete, generate the plots, report, and CSV.

```bash
python -m examples.synthetic_search.report runs/synthetic-search
```

The output is under `runs/synthetic-search/report`. `calibration_comparison.pdf` plots counts by true support size and shows paired differences against the baseline. `synthetic_search_report.pdf` summarizes gains and losses by method and condition. `results.csv` contains per-case fits, fit counts by search phase, best AICc, and the exhaustive check where enabled. The report command checks that all planned cases have finished.

## Run locally

The example results supplied with this benchmark use this 96-case plan.

```bash
python -m examples.synthetic_search.benchmark plan runs/synthetic-local \
    --features 6 10 --support 2 4 6 --replicates 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -m examples.synthetic_search.benchmark run runs/synthetic-local
python -m examples.synthetic_search.report runs/synthetic-local
```

`plan --help` lists the controls for candidate counts, support sizes, signal-to-noise ratios, replicates, sample size, root seed, saliency, and exhaustive verification. Each individual search keeps an explicit set of pending masks, so candidate counts have exponential memory and time costs.

## What is compared

The backward-only baseline fits the full model, runs backward deletion, and examines remaining unexcluded models in descending size. All methods use the same AICc pruning bounds and exhaustive fallback. The 14 experimental variants always run forward searching from the empty mask in descending saliency order. The experimental options are crossed as follows.

| Option | Settings |
| --- | --- |
| Backward search | Off or on. When on, fit the unexcluded one-parameter deletions available at the start of the step and choose the best of those fits as the next current model. |
| Forward pruning | Off, reverse saliency (least salient first), saliency (most salient first), or one-step (fit every unseen immediate supermodel). |
| When to prune forward | At any current size, or only when the current model has more parameters than the best model so far. |

The method IDs start with `B` for backward search on and `N` for backward search off. The `-larger` suffix marks the size gate, evaluated after backward deletion and forward searching. Ordered forward pruning ranks missing parameters using the smallest calibrated proper supermodel containing the current model. The full model's ranking orders any remaining parameters. An ordered extension is fitted when it still has a pending proper submodel, and the chain stops once the available bound fails to reach the current size. One-step pruning tries all immediate supermodels regardless of that bound. Already calibrated models are reused.

## Data and verification

The identifiable generator uses the notebook's centered orthogonal design. The rank-deficient generator samples columns from a smaller orthogonal basis and its pairwise sums, then checks the realized rank. Both use random unit-coefficient supports and centered noise projected off the basis span and scaled to the requested SNR. The benchmark uses the package's linear model, Hessian saliency by default, and `AIC_c` with the number of selected columns as the model parameter count. These controlled designs test structural identifiability under the notebook's noise construction.

Each method starts from its own search state and sees a model's score only when it calibrates that model. The worker shares deterministic fits for computational efficiency. The reported quantity is distinct calibrations per method. At or below `--verify-up-to` (12 by default), the worker separately enumerates every subset, including the empty model, and compares the minimum AICc with every method. Larger cases are checked for agreement among the methods. Numerically tied optimal scores are accepted, including when different supports span the same signal.
