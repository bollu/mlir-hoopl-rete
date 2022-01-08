(cl:in-package #:common-lisp-user)

;;; consider using new-let:
;;; https://github.com/slburson/fset/blob/69c209e6eb15187da04f70ece3f800a6e3cc8639/Code/defs.lisp#L101
(defpackage #:hoopl
  (:use
   #:closer-common-lisp #:fset)
  (:shadowing-import-from :fset
			  ;; Shadowed type/constructor names
			  #:set #:map
			  ;; Shadowed set operations
			  #:union #:intersection #:set-difference #:complement
			  ;; Shadowed sequence operations
			  #:first #:last #:subseq #:reverse #:sort #:stable-sort
			  #:reduce
			  #:find #:find-if #:find-if-not
			  #:count #:count-if #:count-if-not
			  #:position #:position-if #:position-if-not
			  #:remove #:remove-if #:remove-if-not
			  #:substitute #:substitute-if #:substitute-if-not
			  #:some #:every #:notany #:notevery))
