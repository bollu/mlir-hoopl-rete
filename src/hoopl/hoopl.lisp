;;; -*- Mode: Common-Lisp; Author: Siddharth Bhat -*-
;;https://google.github.io/styleguide/lispguide.xml 
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

(defun assert-eq (x y)
  (unless (equal x y)
    (error "expected [~a] == [~a]" x y)))

(defun getter (ty raw-list-ix x)
    (if (equal (first x) ty)
        (if (< raw-list-ix (length x))
            (nth raw-list-ix x)
          (error "expected index [~a] to be valid for [~a]" raw-list-ix x))
      (error "expected type [~a] for ~a" ty x)))

(defparameter *inst-types* (list :assign :add :if :while :goto :bb))

(defun inst->mk-assign (lhs rhs) (list :assign lhs rhs))
(defun inst->mk-add (lhs rhs1 rhs2) (list :add lhs rhs1 rhs2))
(defun inst->mk-if (c body-then body-else) (list :if c body-then body-else))
(defun inst->mk-while (c body) (list  :while c body))
(defun inst->mk-bb (xs) (list :bb xs))
;; (defun inst->mk-goto (lbl) (list :goto lbl))

(defun assign->lhs (x) (getter :assign 1 x))
(defun assign->rhs (x) (getter :assign 2 x))
(defun add->lhs (x) (getter :add 1 x))
(defun add->rhs1 (x) (getter :add 2 x))
(defun add->rhs2 (x) (getter :add 3 x))
(defun bb->body (x) (getter :bb 1 x))

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

;; create fixpoint of const-prop
(defun inst->const-prop-fix (inst env)
  (let* ((res (inst->const-prop inst env)))
    (if (equal inst (result->inst res))
        res ;; then
        (inst->const-prop-fix (result->inst res) (result->env res)) ;; else
        )))


(defun expr->is-const (e)
  "return if expression is constant"
  (numberp e))

(defun result->mk (inst env) (list :result inst env))
(defun result->inst (r) (getter :result 1 r))
(defun result->env (r) (getter :result 2 r))

(defun assign->const-prop (assign env)
  (result->mk assign (acons (assign->lhs assign) (assign->rhs assign) env)))



  ;; https://lispcookbook.github.io/cl-cookbook/clos.html

(defun expr->mk-add (lhs rhs)
  (list :expr->add lhs rhs))
(defun expr-add->lhs (e) (second e))
(defun expr-add->rhs (e) (third e))

(assert (equal (expr-add->lhs (expr->mk-add 1 2)) 1))
(assert (equal (expr-add->rhs (expr->mk-add 1 2)) 2))

(defun expr->eval (x s)
  (cond
    ((numberp x) x)
    ((symbolp x) (cdr (assoc x s)))
    (t (let ((l (expr->eval (expr-add->lhs x) s))
              (r (expr->eval (expr-add->rhs x) s)))
         (if (and (numberp l) (numberp r))
             (+ l r) ;; then return the sum.
             (expr->mk-add l r) ;; else make simplified.
             )))))


(assert-eq (expr->eval 1 nil) 1)
(assert-eq (expr->eval :foo nil) nil)
(assert-eq (expr->eval :foo (acons :foo 10 nil)) 10)
(assert-eq (expr->eval (expr->mk-add 1 2) nil) 3)
(assert-eq (expr->eval (expr->mk-add :x 2) 
                       (acons :x 1 nil)) 3)
(assert-eq (expr->eval (expr->mk-add 2 :x) 
                       (acons :x 1 nil)) 3)
 

(defun add->const-prop (add env)
  (let*
      ((e (expr->mk-add (add->rhs1 add) (add->rhs2 add)))
       (v (expr->eval e env)))
    (format *standard-output* "add->const-prop add: ~a v: ~a" add v)
    (cond 
     ((numberp v) (result->mk (inst->mk-assign (add->lhs add) v) env))
      (t (result->mk add (acons (add->lhs add) v env))))
    ))

(assert-eq (result->inst (add->const-prop (inst->mk-add :x :y :z) nil))
          (inst->mk-add :x :y :z))
(assert-eq (result->inst (add->const-prop (inst->mk-add :x :y :z) nil))
           (inst->mk-add :x :y :z))
(add->const-prop (inst->mk-add :x 1 2) nil)
(assert-eq (result->inst (add->const-prop (inst->mk-add :x 1 2) nil))
           (inst->mk-assign :x 3))
                             

(defun bb->append (bb inst)
  (inst->mk-bb (append (bb->body bb) (list inst))))

       
(reduce (lambda (res x) (list res x)) (list 1 2 3 4) :initial-value 10)
;; constant propagate a basic block by interating on the instructions in the bb.
;; https://jtra.cz/stuff/lisp/sclr/reduce.html
(defun bb->const-prop (bb env)
  (reduce (lambda (res inst)
            (let* ((bb (result->inst res))
                   (env (result->env res))
                   (res (inst->const-prop-fix inst env))
                   (inst (result->inst res))
                   (env (result->env res))
                   (bb (bb->append bb inst)))
              (result->mk bb env)))
          (bb->body bb)
          :initial-value (result->mk (inst->mk-bb nil) env)))

(defun hoopl->run (program niters)
    (inst->const-prop program '()))

(defparameter *program*
  (inst->mk-bb 
   (list (inst->mk-assign :x 1)
         (inst->mk-assign :y 2)
         (inst->mk-add :z :x :y))))

(defparameter *main* (hoopl->run *program* 1))
