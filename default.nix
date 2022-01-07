{ pkgs ? import <nixpkgs> {}}:

with pkgs;

stdenv.mkDerivation {
	name = "mlir-hoopl-rete";
	src = ./src/.;
	nativeBuildInputs = 
		let mlir = pkgs.callPackage ../llvm-project {};
		in [cmake ninja clang lld python3 mlir];
	cmakeFlags = ''
		   -G Ninja  ./
		   -DCMAKE_BUILD_TYPE=RelWithDebInfo 
		   -DLLVM_ENABLE_ASSERTIONS=ON 
		   -DBUILD_SHARED_LIBS=ON
	   	   -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON 
                   -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
		'';
	buildPhase = ''ninja'';
	foo = [emacs ];
}

