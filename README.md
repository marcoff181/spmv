= GPU-computing course 25/26
Evaluation of different SpMV kernels on NVIDIA GPUs.
== How to reproduce
This project uses cmake, if you are on a machine which has a NVIDIA GPU available directly you can either use `just` or take a look at the `justfile` and use those commands. Results will be saved to `results.csv`.
If you are on a computing cluster that uses slurm, simply schedule the provided script with `sbatch sbatch.sh`, it will automatically compile an executable matching the required architecture, and then execute it.
The `results` folder also includes two python scripts to generate statistics about the matrices, and plot the results.
