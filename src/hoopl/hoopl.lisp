;;; -*- Mode: Common-Lisp; Author: Siddharth-Bhat -*-
;;;; https://google.github.io/styleguide/lispguide.xml 
;;;; https://jtra.cz/stuff/lisp/sclr/index.html
;;;; https://lispcookbook.github.io/cl-cookbook/data-structures.html
;;;; https://github.com/bollu/mlir-hoopl-rete/blob/master/reading/hoopl-proof-lerner.pdf
;;;; https://learnxinyminutes.com/docs/common-lisp/
;;;; https://lispcookbook.github.io/cl-cookbook/clos.html
;;;; https://malisper.me/debugging-lisp-part-1-recompilation/
;;;; systems: https://stevelosh.com/blog/2018/08/a-road-to-common-lisp/#s33-systems
;;;; errors and restarts:  https://gigamonkeys.com/book/beyond-exception-handling-conditions-and-restarts.html
;; (defpackage #:hoopl
;;   (:use :closer-mop))

(in-package :hoopl)


(declaim (optimize (debug 3)))

(defun assert-equal (x y)
  (unless (equal x y)
    (error "expected [~a] == [~a]" x y)))

(defclass inst-assign ()
  ((assign-lhs :initarg :lhs :accessor assign-lhs)
   (assign-rhs :initarg :rhs :accessor assign-rhs)))

(defclass inst-add ()
  ((add-lhs :initarg :lhs  :accessor add-lhs)
   (add-rhs1 :initarg :rhs1 :accessor add-rhs1)
   (add-rhs2 :initarg :rhs2 :accessor add-rhs2)))

(defclass inst-if ()
  ((if-cond :initarg :cond :accessor if-cond)
   (if-then :initarg :then :accessor if-then)
   (if-else :initarg :else :accessor if-else)))

(defclass inst-while ()
  ((while-cond :initarg :cond :accessor while-cond)
   (while-body :initarg :body :accessor while-body)))

(defun mk-while (cond_ body) (make-instance 'inst-while cond_ body))

(defclass inst-bb ()
  ((bb-body :initarg :body :accessor bb-body)))

(defun mk-inst-assign (lhs rhs) 
  (make-instance 'inst-assign :lhs lhs :rhs rhs))

(defun mk-inst-add (lhs rhs1 rhs2)
  (make-instance 'inst-add :lhs lhs :rhs1 rhs1 :rhs2 rhs2))

(defun mk-inst-bb (body) (make-instance 'inst-bb :body body))

(defun mk-inst-if (cond_ then else) (make-instance 'inst-if :cond cond_ :then then :else else))

(defgeneric const-prop (i env)
  (:documentation "const propagate the instruction"))

(defun const-prop-fix (i env)
  (let ((res (const-prop i env)))
    (with-slots ((res-i result-inst) (res-env result-env)) res
      (if (equal i res-i)
          res
        (const-prop-fix res-i res-env)))))

(defclass result ()
          ((result-inst :initarg :inst :accessor result-inst)
           (result-env :initarg :env :accessor result-env)))

(defun mk-result (inst env)
  (make-instance 'result :inst inst :env env))

    
(defmethod const-prop ((i inst-assign) env)
  (with-slots ((lhs assign-lhs) (rhs assign-rhs)) i
    (mk-result i (acons lhs rhs env))))

(defclass expr-add ()
  ((expr-add-lhs :initarg :lhs :accessor expr-add-lhs)
   (expr-add-rhs :initarg :rhs :accessor expr-add-rhs)))

(defun mk-expr-add (lhs rhs)
  (make-instance 'expr-add :lhs lhs :rhs rhs))

(defgeneric expr-eval (x s)
  (:documentation "evaluate xpression x at store s"))

(defmethod expr-eval ((x number) s) x)
(assert-equal (expr-eval 1 nil) 1)

(defmethod expr-eval ((x symbol) s) (cdr (assoc x s)))
(assert-equal (expr-eval :foo nil) nil)
(assert-equal (expr-eval :foo (acons :foo 10 nil)) 10)

(defmethod expr-eval ((x expr-add) s)
  (let ((l (expr-eval (expr-add-lhs x) s))
        (r (expr-eval (expr-add-rhs x) s)))
    (if (and (numberp l) (numberp r))
        (+ l r) ; then return the sum.
      (mk-expr-add l r) ; else make simplified.
      )))
(assert-equal (expr-eval (mk-expr-add 1 2) nil) 3)
(assert-equal (expr-eval (mk-expr-add :x 2)
			 (acons :x 1 nil)) 3)
(assert-equal (expr-eval (mk-expr-add 2 :x ) 
			 (acons :x 1 nil)) 3)

(defmethod const-prop ((add inst-add) env)
  (let*
      ((e (mk-expr-add (add-rhs1 add) (add-rhs2 add)))
       (v (expr-eval e env)))
    ;; (format *standard-output* "add->const-prop add: ~a v: ~a" add v)
    (if (numberp v)
        (mk-result (mk-inst-assign (add-lhs add) v) env)
      (mk-result add (acons (add-lhs add) v env)))))


(defclass lattice-top () ())
(defun mk-lattice-top () (make-instance 'lattice-top))

(defgeneric lattice-union (x y)
  (:documentation "take the union of two values in a semilattice"))


;; union for numbers
(defmethod lattice-union ((x number) (y number))
  (if (equal x y)
      x
      (mk-lattice-top)))

(defun akeys (kvs)
  "get keys from an assoc list"
  (mapcar #'car kvs))

;; this is a union for lattice maps, really speaking.
(defun union-assoc-list (xs ys)
  (let ((ks (append (akeys xs) (akeys ys))))
    (mapcar (lambda (k)
	      (let ((xv? (assoc k xs))
		    (yv? (assoc k ys)))
		(cond
		  ((and xv? yv?) (lattice-union xv? yv?))
		  (xv? xv?)
		  (yv? yv?)
		  (t nil))))
	      ks)))
  
(defmethod const-prop ((if_ inst-if) env)
  (let* ((condv (expr-eval (if-cond if_) env)))
  (if (numberp condv)
      (if (equal condv 1)
	  (mk-result (if-then if_) env) ;; condv = 1
	  (mk-result (if-else if_) env)) ;; condv != 1
      (let*
	  ((t-res (const-prop (if-then if_) env))
	   (e-res (const-prop (if-else if_) env)))
	(mk-result (mk-inst-if (if-cond if_)
			       (result-inst t-res)
			       (result-inst e-res))
		   (union-assoc-list
		    (result-env t-res)
		    (result-env e-res)))))))


(defmethod const-prop ((w inst-while) env)
  (error "unimplemented method const-prop for while loop"))

;;;; equivalent upto structure
(defgeneric deepeq (x y))
(defmethod deepeq ((x number) y)
  (equal x y))
(defmethod deepeq ((x symbol) y)
  (eq x y))
                  
(defmethod deepeq (x y)
  (and (equal (class-of x) (class-of y))
       (every (lambda (slot)
                (let* ((name (c2mop:slot-definition-name slot))
                       (xval (slot-value x name))
                       (yval (slot-value y name))
                       (xslotp (slot-boundp x name))
                       (yslotp (slot-boundp y name)))
                  (or (and (not xslotp) ; if x does not have slot bound
                           (not yslotp)) ; then y should not either
                      ; else x has slot bound, so y should as well
                      (and yslotp (deepeq xval yval)))))
              (c2mop:class-slots (class-of x)))))

(assert-equal (deepeq (mk-inst-add :x :y 1) (mk-inst-add :x :y 1)) t)
(assert-equal (deepeq (mk-inst-add :x :y 1) (mk-inst-add :x :x 1)) nil)

(defun assert-deepeq (x y)
  (unless (deepeq x y)
    (error "expected [~a] == [~a]" x y)))

(assert-deepeq (result-inst (const-prop (mk-inst-add :x :y :z) nil))
               (mk-inst-add :x :y :z))
(assert-deepeq (result-inst (const-prop (mk-inst-add :x 1 2) nil))
               (mk-inst-assign :x 3))

(defun bb-append (bb inst)
  (mk-inst-bb (append (bb-body bb) (list inst))))

;;;; constant propagate a basic block by interating on the instructions in the bb.
;;;; https://jtra.cz/stuff/lisp/sclr/reduce.html
(defmethod const-prop ((bb inst-bb) env)
  (reduce (lambda (res inst)
            (let* ((bb (result-inst res))
                   (env (result-env res))
                   (res (const-prop-fix inst env))
                   (inst (result-inst res))
                   (env (result-env res))
                   (bb (bb-append bb inst)))
              (mk-result bb env)))
          (bb-body bb)
          :initial-value (mk-result (mk-inst-bb nil) env)))

(defun hoopl-run (program)
  (const-prop program '()))


(defgeneric debug-show (x))
(defmethod debug-show ((x number)) x)
(defmethod debug-show ((x symbol)) x)
(defmethod debug-show ((xs list)) 
  (mapcar #'debug-show xs))

(defun flatten-list-of-lists (xss) (apply 'concatenate 'list xss))

(defmethod debug-show (x)
  (let*
      ((cls (class-of x))
       (slots (c2mop:class-slots cls))
       (slot-out (loop for slot in slots collect 
                       (let* 
                           ((k (c2mop:slot-definition-name slot))
                            (n (car (c2mop:slot-definition-initargs slot)))
                            (v (slot-value x k)))
                         (list n (debug-show v))))))
    (cons (class-name cls) (flatten-list-of-lists slot-out))))
  
(debug-show 1)
(debug-show :foo)
(debug-show (list 1 2 3))
(debug-show (mk-inst-assign 1 2))

(defparameter *program-assign*
  (mk-inst-bb
   (list (mk-inst-assign :x 1)
         (mk-inst-assign :y 2)
         (mk-inst-add :z :x :y))))
(debug-show *program-assign*)
(defparameter *hoopl-assign* (hoopl-run *program-assign*))

(defparameter *program-if*
  (mk-inst-bb
   (list (mk-inst-assign :x 1)
	 (mk-inst-if
	  :x
	  (mk-inst-bb
	   (list (mk-inst-add :same :x 1)
		 (mk-inst-add :diff :x 3)))
	  (mk-inst-bb
	   (list (mk-inst-add :same :x 1)
		 (mk-inst-add :diff :x -3))))
	 (mk-inst-add :z :same 1)
	 (mk-inst-add :w :diff 2))))
(debug-show *program-if*)
(defparameter *hoopl-if* (hoopl-run *program-if*))
(trace (hoopl-run *program-if*))

