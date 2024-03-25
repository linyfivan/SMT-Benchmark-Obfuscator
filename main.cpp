#include <iostream>
#include <ostream>
#include <set>
#include <type_traits>
#include "z3++.h"
#include <unordered_map>


using namespace z3;
using namespace std;

set<int> archeck = {Z3_OP_EQ, Z3_OP_GE, Z3_OP_LE, Z3_OP_GT, Z3_OP_LT};

void extract(expr const & e, expr_vector* ar_formulae, params p) {
    if (e.is_app()) {
        auto f = e.decl();
        //cout << "application of " << f.name() << f.decl_kind() << ": " << e << "\n";
        if (archeck.find(f.decl_kind()) != archeck.end()) {
            auto e_simp = e.simplify(p);
            //cout << e << endl;
            //cout << e_simp << endl;
            ar_formulae->push_back(e_simp);
        }
        else {
            unsigned num = e.num_args();
            for (unsigned i = 0; i < num; i++) {
                extract(e.arg(i), ar_formulae, p);
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

void exprs_to_matrix(const expr_vector *exprs) {
    for(const auto e : *exprs) {
        
        unordered_map<string ,int> coef_map ;
        
        auto f = (e.decl().decl_kind() == Z3_OP_NOT) ? e.arg(0) : e;
        
        assert(f.num_args() == 2);
        auto monomials =  f.arg(0);
        
        assert(monomials.decl().decl_kind() == Z3_OP_ADD);
        auto constant = f.arg(1);
        for(unsigned i=0; i<monomials.num_args(); i++) {
            auto x = monomials.arg(i);
            //cout<<x<<endl;
            if(x.is_const()) {
                cout<<x<< "is"<<x.decl().name().str()<<endl;
                coef_map[x.decl().name().str()] = 1;
            }
            else {
                assert(x.num_args() == 2);
                assert(x.arg(1).is_const());
                coef_map[x.arg(1).decl().name().str()] = x.arg(0).decl().name().to_int();
                //FIXME:deal with negative number 
            }
        }

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

    //extract(f, a, p);
    cout << *a;
    
}

int main(int argc, char* argv[]) {

    //set params
    context c;
    z3::params p(c);
    p.set("arith_lhs",true);
    //TODO:eliminate ite when reading file
    p.set("elim_ite",true);

    /*read smt2 file, parse only arith terms(><=)

    * origin_exprs: full benchmark
    * arith_exprs: all arth terms of origin_exprs
    */
    auto origin_exprs = c.parse_file(argv[1]);
    expr_vector* arith_exprs = new expr_vector(c);
    for (auto e: origin_exprs) {
        extract(e, arith_exprs, p);
    }


    auto s = origin_exprs.size();
    
    cout << *arith_exprs << endl;
    exprs_to_matrix(arith_exprs);
    //tst_visit();
    return 0;
}