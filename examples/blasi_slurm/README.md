# Blasi model selection on Slurm

Run these commands from the modelSelector checkout with a Python 3.12 environment activated.

```bash
python -m pip install -e . -r examples/blasi_slurm/requirements.txt
python examples/blasi_slurm/run.py --check-inputs
```

Submit a 16-model pilot over the first four motif multipliers.

```bash
sbatch --time=00:30:00 --cpus-per-task=2 examples/blasi_slurm/search.sbatch \
    --output /shared/project/blasi/pilot \
    --supermodel 11110000000000000000000000000000 --n-starts 2
```

Omit `--supermodel` and `--n-starts` for all 32 motif multipliers and 200 optimization starts per fit. The candidate bitmap uses 512 MiB in the full search. The Slurm allocation also needs memory for optimization workers and fitted models.

```bash
sbatch --cpus-per-task=4 --mem=4G examples/blasi_slurm/search.sbatch \
    --output /shared/project/blasi/full
```

`BlasiModel` fits steady-state log residuals with pyPESTO and ranks motif multipliers by `max(H_ii, 0) * abs(log10(multiplier_i))`. Use `--saliency coefficient` to rank by absolute log10 multiplier. The shared basal rate is fitted in every model and is included in ModelSelector's AICc. An omitted motif multiplier is fixed to one.

Each completed fit appends its model ID, AICc score, and named `log10_parameters` to `fits.tsv`. The JSON parameter column contains the fitted motif multipliers and shared basal rate at the best fit. Each value is in log10 coordinates. Omitted motif multipliers are fixed to one.

Resubmit with the same inputs, options, and directory to reuse completed scores. The runner refits the starting model to recover its saliencies, rebuilds the bounds from the saved scores, and continues the search. The order of later fits can change after recovery. `--max-new-fits` limits new scores per submission. 

The default inputs are the [Blasi PEtab benchmark](https://petab-select.readthedocs.io/en/stable/examples/test_case_0009.html), bundled under `inputs/petab/` with its license. Use `--petab-yaml` to supply another Blasi problem or `--data-mat` to override the measurements. The reaction matrix and steady-state reduction are adapted from the supplementary code archive (DOI `10.5281/zenodo.14906119`).
