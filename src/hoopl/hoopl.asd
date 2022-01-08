(defpackage #:hoopl
  (:use :cl :asdf))

(defsystem :hoopl :depends-on (#:closer-mop #:fset)
    :components ((:file "hoopl")))

