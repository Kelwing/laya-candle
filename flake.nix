{
  description = "laya-candle dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f system);
    in
    {
      devShells = forAll (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true; # CUDA toolkit
          };
          base = with pkgs; [ cargo rustc rustfmt clippy rust-analyzer pkg-config openssl ];
          rustSrc = "${pkgs.rustPlatform.rustLibSrc}";
        in
        {
          default = pkgs.mkShell {
            packages = base;
            RUST_SRC_PATH = rustSrc;
          };

          cuda = pkgs.mkShell {
            packages = base ++ [ pkgs.cudaPackages.cudatoolkit pkgs.cudaPackages.cudnn ];
            RUST_SRC_PATH = rustSrc;
            CUDA_ROOT = "${pkgs.cudaPackages.cudatoolkit}";
            LD_LIBRARY_PATH = "${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:/run/opengl-driver/lib";
          };
        });
    };
}
