{
  description = "tensor roguelike — the game is a neural net";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python313;
    in {
      devShells.default = pkgs.mkShell {
        packages = [
          (python.withPackages (ps: with ps; [ numpy tinygrad ]))
        ];
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
        shellHook = ''
          echo "tensor roguelike dev shell — the game is inference"
        '';
      };
    }
  );
}
