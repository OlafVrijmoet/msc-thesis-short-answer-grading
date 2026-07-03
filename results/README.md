# Results

Every experiment run appended one row to a shared tracking CSV: the model
configuration, the dataset and split it ran on, and its metrics (weighted/macro/micro
F1, accuracy, RMSE, Pearson correlation). The final file held 4,871 rows and was lost
with the machine it lived on; it was never committed because the tracking directory
was gitignored during the project.

What survives:

- `performance_tracking_2023-05-31_partial.csv`: a 931-row snapshot recovered from
  the original git history (commit "BERT implemented", 31 May 2023). It covers the
  embedding-based models and the early GPT-3.5 runs, not the final BERT and GPT-3.5
  experiments.
- `notebooks/06_results_analysis.ipynb`: the analysis notebook with its saved outputs,
  which were computed from the complete 4,871-row file in June 2023. The aggregate
  numbers reported in the thesis come from there.
- The thesis itself (`docs/thesis.pdf`) with the full results tables in Appendix A.2.
