;;; -*- Mode: Common-Lisp; Author: Siddharth Bhat -*-
;;https://jtra.cz/stuff/lisp/sclr/index.html
;;https://lispcookbook.github.io/cl-cookbook/data-structures.html
;;https://github.com/bollu/mlir-hoopl-rete/blob/master/reading/hoopl-proof-lerner.pdf
;;https://learnxinyminutes.com/docs/common-lisp/

;; Practically speaking, you should use DEFVAR to
;;  define variables that will contain data you'd want to keep
;; even if you made a change to the source code that uses the variable.
;; For instance, suppose the two variables defined previously are part
;; of an application for controlling a widget factory.
;; It's appropriate to define the *count* 
;; variable with DEFVAR because the number of widgets made so far
;; isn't invalidated just because you make some changes to the widget-making code.

;; https://malisper.me/debugging-lisp-part-1-recompilation/

;; errors and restarts: https://gigamonkeys.com/book/beyond-exception-handling-conditions-and-restarts.html
(declaim (optimize (debug 3)))
(defparameter *size* 50)
(defparameter *inst-types* (list :assign :add :if :while :goto :bb))

(defun inst->mk-assign (lhs rhs) (list :assign lhs rhs))
(defun inst->mk-add (lhs rhs1 rhs2) (list :add lhs rhs1 rhs2))
(defun inst->mk-if (c body-then body-else) (list :if c body-then body-else))
(defun inst->mk-while (c body) (list  :while c body))
(defun inst->mk-bb (xs) (list :bb xs))
;; (defun inst->mk-goto (lbl) (list :goto lbl))

(defun assign->lhs (x) (second x))
(defun assign->rhs (x) (third x))
(defun add->lhs (x) (second x))
(defun add->rhs1 (x) (third x))
(defun add->rhs2 (x) (fourth x))
(defun bb->body (x) (second x))

(defun inst->ty (x)
  "return the type of an instruction"
  (let ((ty (first x)))
    (progn 
      (assert (member ty *inst-types*))
      ty)))

(defun inst->const-prop (x env)
  (case (inst->ty x)
    (:assign (assign->const-prop x env))
    (:add (add->const-prop x env))
    (:if  (if->const-prop x env))
    (:while (while->const-prop x env))
    (:goto (goto->const-prop x env))
    (:bb (bb->const-prop x env))
    (otherwise (error "unknown instruction ~a" x))))
;; https://en.wikipedia.org/wiki/Format_(Common_Lisp)

(defun expr->is-const (e)
  "return if expression is constant"
  (numberp e))

(defun result->mk-rewrite (inst)
  "create a rewrite result"
  (list :result->rewrite inst))


(defun result->mk-propagate (env)
  "create a propagate result"
  (list :result->propagate env))

(defun assign->const-prop (assign env)
  (result->mk-propagate (acons (assign->lhs assign) (assign->rhs assign) env)))

;; ('add lhs rhs1 rhs2)
;; try renaming
(defun add->const-prop (add env)
  (if (and (expr->is-const (add->rhs1 add))
           (expr->is-const (add->rhs2 add)))
      (result->mk-rewrite 
       (inst->mk-assign
        (add->lhs add) 
        (+ (add->rhs1 add) (add->rhs2 add)))) ;; then:  rewrite to cosntant
    (result->mk-propagate (list '+ (add->rhs1 add) (add->rhs2 add))) ;; else: make propagate  
    ))

;; constant propagate a basic block by interating on the instructions in the bb.
;; https://jtra.cz/stuff/lisp/sclr/reduce.html
(defun bb->const-prop (bb env)
  (reduce (lambda (env x) (inst->const-prop x env))
          (bb->body bb)
          :initial-value env))

;; https://gigamonkeys.com/book/loop-for-black-belts.html
(defun hoopl->run (program niters)
  (inst->const-prop program '()))

(defparameter *program*
  (inst->mk-bb 
   (list (inst->mk-assign :x 1)
         (inst->mk-assign :y 2)
         (inst->mk-add :z :x :y))))

(defparameter *main* (hoopl->run *program* 1))
;; TODO: need to build the new program on the side =)
(hoopl->run *program* 1)


