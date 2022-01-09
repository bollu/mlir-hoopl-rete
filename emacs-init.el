;; (evil-mode 1)
(projectile-mode 1)
(lsp-ui-mode 1)
(lsp-mode 1)
(vertico-mode 1)
(company-mode 1)
;; (rainbow-delimiters-mode 1)
(which-key-mode 1)
(show-paren-mode 1)
(projectile-mode 1)
(company-mode 1)
(global-company-mode 1)
(define-key projectile-mode-map (kbd "C-x p") 'projectile-command-map)
(load-theme 'leuven-dark t)

(define-key lisp-mode-map (kbd "C-c C-c") 'sly-compile-defun)
(define-key lisp-mode-map (kbd "C-x C-c") 'sly-compile-defun)
(define-key lisp-mode-map (kbd "C-c C-e") 'sly-eval-defun)
(define-key lisp-mode-map (kbd "C-x C-e") 'sly-eval-defun)
(define-key lisp-mode-map (kbd "C-.") 'sly-edit-definition)
(define-key lisp-mode-map (kbd "C-,") 'sly-pop-definition-stack)
(define-key lisp-mode-map (kbd "C-c >") 'sly-calls-who)
(define-key lisp-mode-map (kbd "C-x >") 'sly-calls-who)
(define-key lisp-mode-map (kbd "C-c <") 'sly-who-calls)
(define-key lisp-mode-map (kbd "C-x <") 'sly-who-calls)

(setq completion-styles '(orderless)
        completion-category-defaults nil
        completion-category-overrides '((file (styles partial-completion))))
(setq vertico-count 20)

(setq inferior-lisp-program "/nix/store/wjaraq6qygaabavbzr71idlif4g5bxwv-sbcl-2.1.11/bin/sbcl")


