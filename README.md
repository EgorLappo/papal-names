## A random-copying model explains the temporal sequence of papal names

Replication code.

### How to run

I use Nix to fix a reproducible development environment. To generate all figures in the paper, first [install Nix](https://nixos.org/download/). Then, `git clone` this repository and `cd` into it. Run `nix develop` to enter the development shell with all necessary dependencies (see [this link](https://nix.dev/concepts/flakes) if Nix complains about experimental features). Finally, call `./run.py` to perform the analyses and make the figures.

As Slatkin's exact tests can take a bit of time, `run.py` tries to use the cached results from the `data/` folder. If you want to rerun all the tests or adjust some parameters, first delete (or move) the `data/` directory.
