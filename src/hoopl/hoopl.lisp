;;; -*- Mode: Common-Lisp; Author: Siddharth Bhat -*-
;;https://google.github.io/styleguide/lispguide.xml 
;;https://jtra.cz/stuff/lisp/sclr/index.html
;;https://lispcookbook.github.io/cl-cookbook/data-structures.html
;;https://github.com/bollu/mlir-hoopl-rete/blob/master/reading/hoopl-proof-lerner.pdf
;;https://learnxinyminutes.com/docs/common-lisp/
;; https://lispcookbook.github.io/cl-cookbook/clos.html
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

(defun assert-equal (x y)
  (unless (equal x y)
    (error "expected [~a] == [~a]" x y)))

(defun getter (ty raw-list-ix x)
    (if (equal (first x) ty)
        (if (< raw-list-ix (length x))
            (nth raw-list-ix x)
          (error "expected index [~a] to be valid for [~a]" raw-list-ix x))
      (error "expected type [~a] for ~a" ty x)))

(defparameter *inst-types* (list :assign :add :if :while :goto :bb))

(defclass inst-assign ()
  ((assign-lhs :initarg :lhs   :accessor assign-lhs)
   (assign-rhs :initarg :rhs :accessor assign-rhs)))

(defclass inst-add ()
  ((add-lhs :initarg :lhs  :accessor add-lhs)
   (add-rhs1 :initarg :rhs1 :accessor add-rhs1)
   (add-rhs2 :initarg :rhs2 :accessor add-rhs2)))

(defclass inst-bb ()
  ((bb-body :initarg :body :accessor bb-body)))

(defun mk-inst-assign (lhs rhs) 
  (make-instance 'inst-assign :lhs lhs :rhs rhs))

(defun mk-inst-add (lhs rhs1 rhs2)
  (make-instance 'inst-add :lhs lhs :rhs1 rhs1 :rhs2 rhs2))

(defun mk-inst-bb (body) (make-instance 'inst-bb :body body))

(defgeneric const-prop (i env) (:documentation "const propagate the instruction"))

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
    (make-instance 'result
                   :result-inst i
                   :result-env (acons lhs rhs env)
                   )))

(defclass expr-add ()
  ((expr-add-lhs :initarg :lhs :accessor expr-add-lhs)
   (expr-add-rhs :initarg :rhs :accessor expr-add-rhs)))

(defun mk-expr-add (lhs rhs)
  (make-instance 'expr-add :lhs lhs :rhs rhs))

(defparameter *e* (make-instance 'expr-add :lhs 1 :rhs 2))

;; To make this a generic method, I'll need to somehow make number
;; and symbol inherit from expr, so that I can say (eval-expr 10 env)
;; or (eval-expr 'x env). How do I do this?
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
        (+ l r) ;; then return the sum.
      (make-instance 'expr-add :lhs l :rhs r) ;; else make simplified.
      )))
(assert-equal (expr-eval (make-instance 'expr-add :lhs 1 :rhs 2) nil) 3)
(assert-equal (expr-eval (make-instance 'expr-add :lhs :x :rhs 2)
                       (acons :x 1 nil)) 3)
(assert-equal (expr-eval (make-instance 'expr-add :lhs 2 :rhs :x) 
                       (acons :x 1 nil)) 3)


(defmethod const-prop ((add inst-add) env)
  (let*
      ((e (mk-expr-add (add-rhs1 add) (add-rhs2 add)))
       (v (expr-eval e env)))
    (format *standard-output* "add->const-prop add: ~a v: ~a" add v)
    (if (numberp v)
        (mk-result (mk-inst-assign (add-lhs add) v) env)
      (mk-result add (acons (add-lhs add) v env)))))

;; equivalent upto structure
(defgeneric deepeq (x y))
(defmethod deepeq ((x number) y)
  (equal x y))
(defmethod deepeq ((x symbol) y)
  (equal x y))
                  
(defmethod deepeq (x y)
  (and (equal (class-of x) (class-of y))
       (every (lambda (slot)
                (let*
                    ((name (slot-definition-name slot))
                     (xval (slot-value x name))
                     (yval (slot-value y name))
                     (xslotp (slot-boundp x name))
                     (yslotp (slot-boundp y name)))
                  (or (and (not xslotp) ;; if x does not have slot bound
                           (not yslotp)) ;; then y should not either
                      ;; else x has slot bound, so y should as well
                      (and yslotp (deepeq xval yval)))))
              (class-slots (class-of x)))))

(assert-equal (deepeq (mk-inst-add :x :y 1) (mk-inst-add :x :y 1)) t)
(assert-equal (deepeq (mk-inst-add :x :y 1) (mk-inst-add :x :x 1)) nil)

(defun assert-deepeq (x y)
  (unless (deepeq x y)
    (error "expected [~a] == [~a]" x y)))

(assert-deepeq (result-inst (const-prop (mk-inst-add :x :y :z) nil))
          (mk-inst-add :x :y :z))
(assert-deepeq (result-inst (const-prop (mk-inst-add :x 1 2) nil))
               (mk-inst-assign :x 3))
                             
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
