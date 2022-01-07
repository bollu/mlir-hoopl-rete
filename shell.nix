{ pkgs ? import <nixpkgs> {} }:
with pkgs;

pkgs.mkShell {
    # nativeBuildInputs is usually what you want -- tools you need to run
    nativeBuildInputs = 
	let emacs-mine = emacs-nox.pkgs.withPackages 
		(epkgs: [epkgs.evil epkgs.lsp-mode epkgs.lsp-ui 
			 epkgs.which-key epkgs.vertico
			 epkgs.company  epkgs.rainbow-delimiters
		         epkgs.slime epkgs.projectile epkgs.leuven-theme]);
	in [emacs-mine sbcl];
	
}
