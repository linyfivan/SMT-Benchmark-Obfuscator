#include <iostream>
#include <ostream>
#include <set>
#include "z3++.h"

using namespace z3;
using namespace std;

set<int> archeck = {Z3_OP_EQ, Z3_OP_GE, Z3_OP_LE, Z3_OP_GT, Z3_OP_LT};

void extract(expr const & e, expr_vector* ar_formulae) {
    if (e.is_app()) {
        auto f = e.decl();
        //cout << "application of " << f.name() << f.decl_kind() << ": " << e << "\n";
        if (archeck.find(f.decl_kind()) != archeck.end()) {
            ar_formulae->push_back(e);
        }
        else {
            unsigned num = e.num_args();
            for (unsigned i = 0; i < num; i++) {
                extract(e.arg(i), ar_formulae);
            }
        }
    }
    /* 
    else if (e.is_quantifier()) {

    }
    */
    else { 
        assert(e.is_var());
    }
}

void tst_visit() {
    cout << "visit example\n";
    context c;

    expr x = c.int_const("x");
    expr y = c.int_const("y");
    expr z = c.int_const("z");
    expr f = x*x - y*y >= 0;
    expr_vector* a = new expr_vector(c);

    extract(f, a);
    cout << *a;
}

int main() {
    context c;

    auto origin_exprs = c.parse_file("test.smt2");
    expr_vector* arith_exprs = new expr_vector(c);

    for (auto e: origin_exprs) {
        extract(e, arith_exprs);

    }
    auto s = origin_exprs.size();
    
    
    cout << *arith_exprs;
    //tst_visit();
    return 0;
}