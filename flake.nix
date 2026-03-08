{
  description = "tensor roguelike — the game is a neural net";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python313.withPackages (ps: with ps; [ numpy tinygrad ]);
    in {
      packages.default = pkgs.writeShellScriptBin "roguelike" ''
        CPU=1 exec ${python}/bin/python ${self}/roguelike.py "$@"
      '';
      apps.default = { type = "app"; program = "${self.packages.${system}.default}/bin/roguelike"; };
      devShells.default = pkgs.mkShell {
        packages = [ python ];
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ];
        shellHook = ''echo "tensor roguelike — run with: CPU=1 python roguelike.py"'';
      };
    }
  );
}
