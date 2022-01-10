;;; -*- Mode: Common-Lisp; Author: Siddharth-Bhat -*-
;;;; Version 1:
;;;; ---------
;;;; Keep the project at ~/quicklisp/local-projects/hoopl.asdl. Run M-x slime. Follow with (ql:quickload hoopl) in REPL. Finally
;;;; switch to hoopl.lisp and type C-c ~ [slime-sync-package-and-default-directory]
;;;; Version 2:
;;;; ---------
;;;; Open hoopl.asd, run C-c C-k [compile file]. Switch to REPL, then run (ql:quickload hoopl). Finally switch to hoopl.lisp
;;;; and type C-c ~ [slime-sync-package-and-default-directory] to enter the hoopl module in the repl

(in-package :hoopl)
;; (sb-ext:restrict-compiler-policy 'debug 3 3)
(declaim (optimize (speed 0) (space 0) (debug 3)))

(defun assert-equal (x y)
  (unless (equal x y)
    (error "expected ~%[~a] == ~%[~a]" x y)))

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

(defclass inst-nop () ())

(defclass inst-bb ()
  ((bb-body :initarg :body :accessor bb-body)))

(defun mk-inst-assign (lhs rhs) 
  (make-instance 'inst-assign :lhs lhs :rhs rhs))

(defun mk-inst-add (lhs rhs1 rhs2)
  (make-instance 'inst-add :lhs lhs :rhs1 rhs1 :rhs2 rhs2))

(defun mk-inst-bb (body) (make-instance 'inst-bb :body body))

(defun mk-inst-if (cond_ then else) (make-instance 'inst-if :cond cond_ :then then :else else))

(defun mk-inst-while (cond_ body) (make-instance 'inst-while :cond cond_ :body body))

(defun mk-inst-nop () (make-instance 'inst-nop))
(defgeneric const-prop (i env)
  (:documentation "const propagate the instruction"))

(defun const-prop-fix (i env)
  (let ((res (const-prop i env))) ;
    (with-slots ((res-i result-inst) (res-env result-env)) res
      (if (equal (debug-show i) (debug-show res-i))
          res
        (const-prop-fix res-i env)))))

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
(defmethod debug-show ((x lattice-top)) :LATTICE-TOP-VAL)

(defclass lattice-bot() ())
(defun mk-lattice-bot () (make-instance 'lattice-bot))
(defmethod debug-show ((x lattice-bot)) :LATTICE-BOT-VAL)

(defgeneric lattice-union (x y)
  (:documentation "take the union of two values in a semilattice"))


;; union for numbers
(defmethod lattice-union ((x number) (y number))
  (if (equal x y)
      x
      (mk-lattice-top)))

(defmethod lattice-union ((x expr-add) y)
  (if (deepeq x y)
      x
      (mk-lattice-top)))

(defun akeys (kvs)
  "get keys from an assoc list"
  (mapcar #'car kvs))


(defmethod lattice-union ((x lattice-bot) y) y)
(defmethod lattice-union (x (y lattice-bot)) x)
(defmethod lattice-union ((x lattice-top) y) (mk-lattice-top))
(defmethod lattice-union (x (y lattice-top)) (mk-lattice-top))



(defun env@ (e k)
  (let ((v? (cdr (assoc k e))))
    (if v? v? (mk-lattice-bot))))

(env@ (list (cons :a 1) (cons :b 2)) :a)
(if (cdr (assoc :a (list (cons :a 1) (cons :b 2)))) 10 20)

;; this is a union for lattice maps, really speaking.
(defun env-union(xs ys)
  (let ((ks (remove-duplicates (append (akeys xs) (akeys ys)))))
    (mapcar (lambda (k)
	      (cons k (lattice-union (env@ xs k) (env@ ys k))))
	    ks)))


;; union works fine.
(let ((e1 (list (cons :a 0) (cons :b 2) (cons :x 20)))
      (e2 (list (cons :a 0) (cons :b 3) (cons :y 10))))
  (env-union e1 e2))
  
(defmethod const-prop ((if_ inst-if) env)
  (let* ((condv (expr-eval (if-cond if_) env)))
    (if (numberp condv)
	(if (equal condv 1)
	    (mk-result (if-then if_) env) ;; condv = 1
	    (mk-result (if-else if_) env)) ;; condv != 1
	(let*
	    ((t-res (const-prop-fix (if-then if_) env))
	     (e-res (const-prop-fix (if-else if_) env)))
	  (mk-result (mk-inst-if (if-cond if_)
				 (result-inst t-res)
				 (result-inst e-res))
		     (env-union
		      (result-env t-res)
		      (result-env e-res)))))))


;; x <= y iff x \/ y = y 
;; (defgeneric lattice-leq (l r))
;; (defmethod lattice-leq ((l number) (r lattice-top)) t)
;; (defmethod lattice-leq ((l lattice-top) r) (eq r (mk-lattice-top)))
;; (defmethod lattice-leq ((l number) (r number)) (eq l r))

(defmethod debug-show  ((x number)) x)
(defun lattice-leq (l r)
  (equal (debug-show (lattice-union l r)) (debug-show r)))


(assert-equal (lattice-leq (mk-lattice-bot) (mk-lattice-top)) t)
(assert-equal (lattice-leq (mk-lattice-top) (mk-lattice-bot)) nil)
(assert-equal (lattice-leq 1 1) t)
(assert-equal (lattice-leq 1 2) nil)
(assert-equal (lattice-leq 1 (mk-lattice-top)) t)
(assert-equal (lattice-leq 1 (mk-lattice-bot)) nil)
(assert-equal (lattice-leq (mk-lattice-bot) 1) t)

;; check if left <= right for environments
(defun env-leq (left right)
  (every
   (lambda (k)
     (lattice-leq (env@ left k)
		  (env@ right k)))
   (akeys (append left right))))




(defmethod const-prop ((w inst-while) env)
  (let* ((r-body (const-prop-fix (while-body w) env)))
    (if (equal (debug-show (while-body w))
	       (debug-show (result-inst r-body)))
	(mk-result w env) ;; nothing was changed.
	;; else, upgrade loop
	;; we need to propagate our new proposed environment over the OLD Loop?
	(let* ((r-oo (const-prop-fix (while-body w) (result-env r-body))))
	  (if (env-leq (result-env r-oo) (result-env r-body))
	      (mk-result (mk-inst-while (while-cond w)
					(result-inst r-oo))
			 (result-env r-oo))
	      (mk-result w (result-env r-body)))))))

;; DESIGN FOR BETTER UI/UX
;; (? x y z)  -> if
;; (@ e k) -> lookup
;; (@ e k v) -> update
;; (@? e k) -> t/f if exists.
;; ($ f x) -> application
;; (\ (x y z) -> lambda
;; (# x) -> pattern match on x



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
  (const-prop-fix program '()))


(defgeneric debug-show (x))
(defmethod debug-show ((x number)) x)
(defmethod debug-show ((x symbol)) x)
(defmethod debug-show ((xs list)) 
  (mapcar #'debug-show xs))

(defun flatten-list-of-lists (xss) (apply 'concatenate 'list xss))

(defmethod debug-show (x)
  (let*
      ((cls (class-of x))
       ;; (slots (c2mop:class-slots cls))
       (slots (class-slots cls))
       (slot-out (loop for slot in slots collect 
                       (let* 
                           (;; (k (c2mop:slot-definition-name slot))
                            (k (slot-definition-name slot))
                            ;; (n (car (c2mop:slot-definition-initargs slot)))
                            (n (car (slot-definition-initargs slot)))
                            (v (slot-value x k)))
                         ;; (list n (debug-show v)
			 (list (debug-show v))
			 ))))
    (cons (class-name cls) (flatten-list-of-lists slot-out))))

(defmethod print-object ((x inst-while) stream)
  (format stream "~a" (debug-show x)))

(defmethod print-object ((x result) stream)
  (format stream "~a"
	  (list :result
		(debug-show (result-inst x))
		(result-env x))))

(defmethod print-object ((x inst-bb) stream)
  (format stream "~s" (debug-show x)))




(debug-show 1)
(debug-show :foo)
(debug-show (list 1 2 3))
(debug-show (mk-inst-assign 1 2))

(defparameter *assign*
  (mk-inst-bb
   (list (mk-inst-assign :x 1)
         (mk-inst-assign :y 2)
         (mk-inst-add :z :x :y))))
(defparameter *hoopl-assign* (hoopl-run *assign*))
(debug-show (result-inst *hoopl-assign*))
(result-env *hoopl-assign*)
;; check that environment has z value correctly
(assert-equal (list (cons :z 3) (cons :y 2) (cons :x 1))
	      (result-env *hoopl-assign*))

(assert-equal
 (debug-show (result-inst *hoopl-assign*))
 (debug-show (mk-inst-bb
   (list (mk-inst-assign :x 1)
         (mk-inst-assign :y 2)
         (mk-inst-assign :z 3)))))
	       

(defparameter *if-const-cond*
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
;; (debug-show *if-const-cond*)
(defparameter *hoopl-if-const-cond* (hoopl-run *if-const-cond*))
(debug-show (result-inst *hoopl-if-const-cond*))
(assert-equal
 (debug-show (result-inst *hoopl-if-const-cond*))
  (debug-show (mk-inst-bb
   (list (mk-inst-assign :x 1)
	  (mk-inst-bb
	   (list (mk-inst-assign :same 2) ;; x + 1 = 2
		 (mk-inst-assign :diff 4))) ;; x + 3 = 4
	 (mk-inst-assign :z 3) ;; same + 1 = 2 + 1 = 3
	 (mk-inst-assign :w 6))))) ;; diff + 2 = 4 + 2 = 6
	       



(defparameter *if-var-cond*
  (mk-inst-bb
   (list (mk-inst-if
	  :x
	  (mk-inst-bb
	   (list (mk-inst-assign :same 1)
		 (mk-inst-assign :diff 3)))
	  (mk-inst-bb
	   (list (mk-inst-assign :same 1)
		 (mk-inst-assign :diff  -3))))
	 (mk-inst-add :z :same 1)
	 (mk-inst-add :w :diff 2))))
(debug-show *if-var-cond*)

(defparameter *hoopl-if-var-cond* (hoopl-run *if-var-cond*))
(debug-show (result-inst *hoopl-if-var-cond*))
(assert-equal
 (debug-show (result-inst *hoopl-if-var-cond*))
 (debug-show (mk-inst-bb
   (list (mk-inst-if
	  :x
	  (mk-inst-bb
	   (list (mk-inst-assign :same 1)
		 (mk-inst-assign :diff 3)))
	  (mk-inst-bb
	   (list (mk-inst-assign :same 1)
		 (mk-inst-assign :diff  -3))))
	 (mk-inst-assign :z 2) ;; same + 1
	 (mk-inst-add :w :diff 2)))))

(defparameter *program-while-speculation-succeeds*
  (mk-inst-bb
   (list (mk-inst-assign :x 1) 
	 (mk-inst-while
	  :cond
	  (mk-inst-if
	   :x
	   (mk-inst-assign :ifval 10)
	   (mk-inst-add :x :x 1))))))


(debug-show *program-while-speculation-succeeds*)
(hoopl-run *program-while-speculation-succeeds*)

(defparameter *hoopl-while-speculation-succeeds*
  (hoopl-run *program-while-speculation-succeeds*))

;; (debug-show (result-inst *hoopl-while-speculation-succeeds*))
(assert-equal
 (debug-show (result-inst *hoopl-while-speculation-succeeds*))
 (debug-show (mk-inst-bb
	      (list (mk-inst-assign :x 1)
		    (mk-inst-while
		     :cond
		     (mk-inst-assign :ifval 10))))))

(defparameter *program-while-speculation-fails*
  (mk-inst-bb
   (list (mk-inst-assign :x 1) 
	 (mk-inst-while
	  :cond-while
	   (mk-inst-add :x :x 1)))));; assignment should be simplified
;; (debug-show *program-while-speculation-fails*)
(defparameter *hoopl-while-speculation-fails*
   (hoopl-run *program-while-speculation-fails*))
(debug-show (result-inst *hoopl-while-speculation-fails*))
(assert-equal
 (debug-show (result-inst *hoopl-while-speculation-fails*))
 (debug-show (mk-inst-bb
	      (list (mk-inst-assign :x 1)
		    (mk-inst-while
		     :cond-while
		     (mk-inst-add :x :x 1))))))

