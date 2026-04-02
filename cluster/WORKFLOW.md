# DLC Cluster Workflow Guide

This document outlines the standard day-to-day workflow for running thesis experiments on the university DLC cluster using Slurm and Pyxis/Enroot.

## 1. Daily Workflow (Running Experiments)
You do **not** need to rebuild the container when you change your Python code. Your code is mounted live into the container at runtime.

**Step 1: Write code locally**
Edit your Python scripts and Slurm `.sub` files on your local machine.

**Step 2: Sync to the cluster**
Run this from your local Mac terminal (in the root `Thesis` directory):
```bash
bash cluster/sync_to_cluster.sh
```

**Step 3: SSH and Submit**
Connect to the cluster and submit your job:
```bash
ssh user@cluster
# (Once logged in to login01):
sbatch ~/thesis/cluster/your_experiment.sub
```

---

## 2. Updating Dependencies (Rebuilding Container)
If you add a new package to `requirements.txt`, you *must* rebuild the `.sqsh` container so the package is installed inside it.

1. **Local:** Update `requirements.txt` and run `bash cluster/sync_to_cluster.sh`.
2. **Local:** SSH into the cluster (`ssh user@cluster`).
3. **Cluster (Login node):** Request an interactive compute node:
   ```bash
   srun -p dlc --pty --time=01:00:00 --mem=16G --cpus-per-task=4 /bin/bash
   ```
4. **Cluster (Compute node):** Run the setup script to overwrite the container:
   ```bash
   bash ~/thesis/cluster/setup_container.sh
   ```
5. **Cluster:** Type `exit` to leave the compute node. You are now ready to run jobs with the new packages.

---

## 3. Slurm Commands Cheat Sheet

- **Submit a job:** `sbatch <file.sub>`
- **Check your jobs:** `squeue -u $USER`
- **Cancel a job:** `scancel <JOB_ID>`
- **View live output:** `tail -f <job-name-JOB_ID.out>`