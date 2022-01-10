(cl:in-package #:asdf-user)

;; 23:49 <@jackdaniel> bollu: I don't know whether this is documented
(setf asdf:*compile-file-warnings-behaviour* :error)

;; https://lispcookbook.github.io/cl-cookbook/systems.html
(asdf:defsystem :hoopl
  :depends-on (:closer-mop :fset)
  :components
  ((:file "packages")
   (:file "hoopl")))

