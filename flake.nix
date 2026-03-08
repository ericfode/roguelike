{
  description = "tensor roguelike — the game is a neural net";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      python = pkgs.python313.withPackages (ps: with ps; [ numpy tinygrad ]);
      cuda = pkgs.cudaPackages;
      cudart-lib = cuda.cuda_cudart.override { };
    in {
      packages.default = pkgs.writeShellScriptBin "roguelike" ''
        export LD_LIBRARY_PATH="/usr/lib/wsl/lib:''${LD_LIBRARY_PATH:-}"
        exec ${python}/bin/python ${self}/main.py "$@"
      '';
      packages.classic = pkgs.writeShellScriptBin "roguelike-classic" ''
        CPU=1 exec ${python}/bin/python ${self}/roguelike.py "$@"
      '';
      apps.default = { type = "app"; program = "${self.packages.${system}.default}/bin/roguelike"; };
      apps.classic = { type = "app"; program = "${self.packages.${system}.classic}/bin/roguelike-classic"; };
      devShells.default = pkgs.mkShell {
        packages = [
          python
          cuda.cuda_nvcc
          cuda.cuda_cudart
        ];
        LD_LIBRARY_PATH = "/usr/lib/wsl/lib:"
          + pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
        shellHook = ''echo "tensor roguelike — GPU: $(python -c 'from tinygrad import Device; print(Device.DEFAULT)' 2>/dev/null || echo 'detecting...')"'';
      };
    }
  );
}
