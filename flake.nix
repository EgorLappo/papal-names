{
  description = "papal names";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    ewens-sampler = {
      url = "github:EgorLappo/ewens-sampler";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    ewens-configurations = {
      url = "github:EgorLappo/ewens-configurations/main";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ewens-sampler,
      ewens-configurations,
      ...
    }:

    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs { inherit system; };

        python-env = pkgs.python3.withPackages (
          ps: with ps; [
            tqdm
            joblib
            numpy
            pandas
            polars
            pyarrow
            scipy
            matplotlib
            seaborn
            jupyter
            sympy
          ]
        );

        es = ewens-sampler.defaultPackage.${system};
        ec = pkgs.callPackage ewens-configurations { };

      in
      {
        devShells.default =
          with pkgs;
          mkShell {
            name = "pope";
            buildInputs = [
              # automation
              just

              # formatter
              ruff

              # environments
              python-env

              # sampler
              es
              # enumerator
              ec
            ];

            shellHook = ''
              export PYTHONPATH="${python-env}/bin/python"
              alias py="${python-env}/bin/python"
              ${pkgs.kittysay}/bin/kittysay -t "mreow"
            '';
          };
      }
    );
}
