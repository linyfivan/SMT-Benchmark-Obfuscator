(set-info :smt-lib-version 2.6)
(set-logic QF_LIA)
(set-info :source |
Older mathsat benchmarks.  Contact Mathsat group at http://mathsat.itc.it/ for
more information.

This benchmark was automatically translated into SMT-LIB format from
CVC format using CVC Lite
|)
(set-info :category "industrial")
(set-info :status sat)
(declare-fun X_1 () Int)
(declare-fun X_2 () Int)
(declare-fun X_3 () Int)
(assert (>= (+ X_1 X_3 X_2 67) 0))
(assert (>= (+ X_1 (+ X_3 X_2) 67) 0))
(assert (> 9 (+ X_1 (+ 8 (* 2 X_3)) X_2 67)))
(assert (> (+ X_1 X_3) (+ X_1 (+ 8 (* 2 X_3)) X_2 67)))

(check-sat)
(exit)