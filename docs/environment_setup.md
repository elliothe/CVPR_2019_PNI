# Environment setup with Anaconda

## Exporting the conda environment
The environment export follows the introductions provided in :
https://stackoverflow.com/questions/41274007/anaconda-export-environment-file

```bash
conda env export | grep -v "^prefix: " > environment.yml
```

## Create the environment
First, changing the environment name in `environment.yml` to your preferred name.

Either way, the other user then runs:
```bash
conda env create -f environment.yml
```
and the environment will get installed in their default conda environment path.

If you want to specify a different install path than the default for your system (not related to 'prefix' in the environment.yml), just use the -p flag followed by the required path.
```bash
conda env create -f environment.yml -p /home/user/anaconda3/envs/env_name
```