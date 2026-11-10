## Quo nomine vis vocari? Random-copying, conformity, and anti-conformity in the temporal sequence of papal names

### Quick guide on how to run

I use Nix to fix a reproducible development environment. To generate all figures in the paper, first [install Nix](https://nixos.org/download/). Then, `git clone` this repository and `cd` into it. Run `nix develop` to enter the development shell with all necessary dependencies (see [this link](https://nix.dev/concepts/flakes) if Nix complains about experimental features).

To execute the analyses and generate figures, call each figure script in order after entering the development shell:
```bash
nix develop
python figure_1_2_3.py
python figure_4.py
python figure_s1.py
```

As Slatkin's exact tests can take a bit of time, each script tries to use the cached results from the `data/` folder. If you want to rerun all the tests or adjust some parameters, first delete (or move) the `data/` directory.

The supplementary figure tests take the most time to generate, so I provide a separate script that *just* performs the exact tests for the power calculation without plotting code (`figure_s1_compute.py`), together with an example of how to launch the computation on an HPC cluster (`slurm/s1.sbatch`).

### Paper metadata

**PAPER TITLE:** Quo nomine vis vocari? Random-copying, conformity, and anti-conformity in the temporal sequence of papal names

**ABSTRACT:** The study of cultural evolution seeks to understand the processes by which behavioral variants are chosen in human cultures over time. The selection of new popes, each of whom chooses a papal name --- typically reusing previous names in reference to previous popes --- is among the longest ongoing cultural processes taking place in a single human institution. Here, we test the papal naming sequence against predictions of models developed in population genetics and stochastic processes --- Ewens sampling theory and the Chinese restaurant process --- which in the case of papal names amount to randomly copying an existing name in proportion to its frequency, with the possibility of innovation. Although papal name choices are careful individual decisions, we find that the sequence of papal names often accords with a random-copying model, with long periods in which a model of conformity in name choices more closely fits the name sequence and other periods in which anti-conformity provides a closer fit. Hence, despite the consideration that enters into individual choices, aggregate cultural behavior in a 2000-year old human process can potentially be described with simple laws. Our methods can potentially be used to deepen quantitative analyses of conformity and anti-conformity in other cultural processes that involve discrete cultural variants.

**PUBLICATION INFORMATION:** [https://arxiv.org/abs/2605.07028](https://arxiv.org/abs/2605.07028)

**GENERAL INFORMATION**

1. Title of Datasets: [`popes-list.csv`](popes-list.csv).

2. Author information
  * Egor Lappo, Department of Biology, Stanford University, elappo@stanford.edu
  * Noah A. Rosenberg, Department of Biology, Stanford University

3. Date of data collection: 2026-05-07

4. License: CC-BY (see [LICENSE](LICENSE))

5. Zenodo DOIs. This repository: 10.5281/zenodo.23617872; sampling program: 10.5281/zenodo.23617872

DATA & FILE OVERVIEW

1. File List:

*Data*

`popes-list.csv`: `csv` file containing the information on Catholic popes and reign start dates.

*Code*

`figure_1_2_3.py`: Python script to perform Slatkin Exact tests and produce Figures 1, 2, 3.

`figure_4.py`: Python script to perform Slatkin Exact tests and protuce Figure 4.

`figure_s1.py`: Python script to perform power calculation for the Slatkin Exact test and produce Figure S1.

2. Relationship between files, if important: N/A

3. Additional related data collected that was not included in the current data package: N/A

4. Are there multiple versions of the dataset? No

METHODOLOGICAL INFORMATION:

1. Description of methods used for collection/generation of data: see [arxiv preprint](https://arxiv.org/abs/2605.07028]) above.

2. Methods for processing the data: Python scripts described above were used to process and analyze tha data. 

3. Instrument- or software-specific information (including package versions) needed to interpret the data: see [`flake.lock`](flake.lock).

DATA-SPECIFIC INFORMATION FOR: 

`popes-list.csv`

1. Number of variables: 3

2. Number of cases/rows: 266

3. Variable List:

* `pope`: full name of a pope
* `name`: first name of a pope, latin, normalized
* `start`: year in which a pope started their reign

4. Missing data codes: NA
