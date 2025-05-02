# LaGP


## SIMULATIONS

- Generate data: Check the "config_data_generation.yaml" file, then run from command line (from LaGP directory) "python scripts/gen_data.py --config config_data_generation.yaml". The resulting data will be saved into the folder "data/simulation".

- Run (python) simulation: Check the "config_run_simulation.yaml" file, then run from command line (from LaGP directory) "python scripts/run_sim.py --config config_run_generation.yaml". The results will be saved into the folder "results/simulation" as "simulation_results_python.pkl".

- Run (R) simulation: Execute the R script "sim.R". The results will be save into the folder "results/simulation" as "simulation_results_R.pkl".

- Merge simulation results: Run from command line (from LaGP directory) "python scripts/merge.py". The results will be saved into the folder "results/simulation" as "simulation_results.pkl".

- Analyse results: Check the "config_results_simulation.yaml" file, then run from command line (from LaGP directory) "python scripts/analyse_sim.py". The results will be saved into the folder "results/simulation". In particular the latex table will be save there under "table_results.tex", while the plots will be saved inside the "images" folder.



## REAL-WORLD DATA
 - Run (python) simulation: Check the "config_run_real.yaml" file, then run from command line (from LaGP directory) "python scripts/run_real.py --config config_run_real.yaml". The results will be saved into the folder "results/realdata" as "realdata_results_python.pkl".

- Run (R) simulation: Execute the R script "real.R". The results will be save into the folder "results/realdata" as "realdata_restuls_R.pkl".

- To do: implement new merge and analyse script.