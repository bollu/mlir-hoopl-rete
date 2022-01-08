(cl:in-package #:asdf-user)

;; https://lispcookbook.github.io/cl-cookbook/systems.html
(asdf:defsystem :hoopl
  :depends-on (:closer-mop :fset)
  :components
  ((:file "packages")
   (:file "hoopl")))
