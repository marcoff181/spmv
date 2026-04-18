{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:

    let
      system = "x86_64-linux";

      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {

        packages = with pkgs; [
          python313
          python313Packages.scipy

          pkg-config
          openblas
          gcc
          just
          cmake
          cudaPackages.cudatoolkit
        ];

        shellHook = ''
          export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
        '';

      };
    };
}
