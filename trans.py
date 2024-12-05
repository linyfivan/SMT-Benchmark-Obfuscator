from sympy.polys.polyoptions import Symbols
from timeit import default_timer as timer
from pyformlang.finite_automaton import EpsilonNFA, State, Symbol, Epsilon
from z3.z3util import get_vars
import z3
import sympy as sp
import re
import numpy as np
import ast
import os, sys
import random
import warnings
import string
import logging
import sre_constants
import sre_parse
import typing
import time
import pynini as pn
from pynini.lib import byte
from pynini.lib import pynutil
from dataclasses import dataclass
from enum import Enum
import sys



def show_matrix_info(m, mname):
    m_64 = np.array(m).astype(np.float64)
    #print('\n=============== ' + mname + ' info ================\n')
    #print(m)
    #print('shape of ' + mname + ': ', m.shape)
    #print('rank of ' + mname + ': ', np.linalg.matrix_rank(m_64))
    try:
        print('det of ' + mname + ': ', np.linalg.det(m_64))
    except:
        pass

def rand_matrix_gen(m, n=None, min_num=1, max_num=10, invertible=True):
    if n is None:
        n = m
    ret = sp.randMatrix(m, n, min = min_num, max=max_num)
    ret_64 = np.array(ret).astype(np.float64)
    ret_64_inv = None
    if invertible:
        while ret_64_inv is None:
            try:
                if np.linalg.cond(ret_64) < 1/np.finfo(ret_64.dtype).eps:
                    ret_64_inv = np.linalg.inv(ret_64)
                else:
                    ret = sp.randMatrix(m, n, min = min_num, max=max_num)
                    ret_64 = np.array(ret).astype(np.float64)
                    ret_64_inv = None
            except:
                ret = sp.randMatrix(m, n, min = min_num, max=max_num)
                ret_64 = np.array(ret).astype(np.float64)
    return ret, ret_64_inv

def uni_matrix_gen(m, min_num=-10, max_num=10):
    '''
    unimodular matrix: integer matrix whose det is +-1, has property that
    its inverse is also an integer matrix
    '''
    ret = sp.randMatrix(m, min = min_num, max=max_num)
    for i in range(m):
        ret[i, m-i-1] = 1
        for j in range(m-i-1):
            ret[i, j] = 0
    ret_64_inv = np.linalg.inv(np.array(ret).astype(np.float64))
    return ret, ret_64_inv

def rand_vector_gen(length):
    res = []
    for i in range(length):
        res.append(random.randint(1, 100))
    return res

def exprs_get_arith_vars(exprs):
    var_lst = set()
    for expr in exprs:
        for var in get_vars(expr):
            if z3.is_arith(var):
                var_lst.add(var)
    return list(var_lst)

def m_substitution(exprs, vars, Q, r):
    sub = []
    for i in range(len(vars)):
        new_var = r[i]
        for j in range(len(vars)):
            new_var = new_var + Q[i, j] * vars[j]
        #print('new var' , i, ': ', new_var)
        sub.append((vars[i], new_var))
    new_exprs = []
    for expr in exprs:
        new_exprs.append(z3.simplify(z3.substitute(expr, sub), arith_lhs=True))
    return new_exprs

def transform_smt2(filename):
    c = z3.Context()
    s1 = z3.Solver(ctx=c)
    exprs = z3.parse_smt2_file(filename, ctx=c)
    s1.add(exprs)
    #print('expressions:\n', exprs)

    vars = exprs_get_arith_vars(exprs)
    #if (len(vars) > 100):
    #    vars = vars[0:100]
    #print("vars:", vars)
    Q, Q_inv = uni_matrix_gen(len(vars))
    r = rand_vector_gen(len(vars))
    new_exprs = m_substitution(exprs, vars, Q, r)

    s1.reset()
    s1.add(new_exprs)
    #print('new expressions:\n', new_exprs)
    return s1.to_smt2()

pn.default_token_type('symbol')
ascii_table = pn.SymbolTable()
ascii_table.add_symbol("<eps>", 0)
for i in range(12, 128):
  ascii_table.add_symbol(chr(i), i)

def draw(fst, symbol_table=ascii_table):
  fst.set_input_symbols(symbol_table)
  fst.set_output_symbols(symbol_table)
  return fst

def gen_rand_fst_by_state(sigma, state = 6, seed = 5):
    random.seed(seed)
    f = pn.Fst()
    s = []
    for i in range(state):
        s.append(f.add_state())
    f.set_start(s[0])
    for i in range(state-1):
        p1 = random.sample(sigma, len(sigma))
        p2 = random.sample(sigma, len(sigma))
        p3 = random.sample(sigma, len(sigma))
        for j in range(len(sigma)):
            f.add_arc(s[i], pn.Arc(ord(sigma[j]), ord(p1[j]), 0, s[i+1]))
        f.set_final(s[i])
    f.set_final(s[state-1])
    f.add_arc(s[state-1], pn.Arc(0, 0, 0, s[0]))
    return f

def gen_rand_len_fst_by_state(sigma, state = 3, seed = 5, translate = 5, scale = 2):
    random.seed(seed)
    f = pn.Fst()
    s = []
    for i in range(state):
        s.append(f.add_state())
    f.set_start(s[0])
    
    if (scale > 1):
        sc = []
        for i in range((scale-1)*(state-1)):
            sc.append([f.add_state() for _ in range(len(sigma))])
        for i in range(state-1):
            p1 = random.sample(sigma, len(sigma))
            p2 = random.sample(sigma, len(sigma))
            for j in range(len(sigma)):
                f.add_arc(s[i], pn.Arc(ord(sigma[j]), ord(p1[j]), 0, sc[i*(scale-1)][j]))
                f.add_arc(sc[(i+1)*(scale-1)-1][j], pn.Arc(0, ord(p2[j]), 0, s[i+1]))
            for j in range(scale-2):
                p1 = random.sample(sigma, len(sigma))
                for k in range((len(sigma))):
                    f.add_arc(sc[i*(scale-1)+j][k], pn.Arc(0, ord(p1[k]), 0, sc[i*(scale-1)+j+1][k]))
            f.set_final(s[i])
    else:
        for i in range(state-1):
            p1 = random.sample(sigma, len(sigma))
            for j in range(len(sigma)):
                f.add_arc(s[i], pn.Arc(ord(sigma[j]), ord(p1[j]), 0, s[i+1]))
            f.set_final(s[i])
    f.set_final(s[state-1])

    if (translate > 0):
        p2 = ''.join([random.choice(sigma) for _ in range(translate)])
        t = []
        for i in range(translate):
            t.append(f.add_state())
        for i in range(translate-1):
            f.add_arc(t[i], pn.Arc(0, ord(p2[i]), 0, t[i+1]))
        f.add_arc(t[translate-1], pn.Arc(0, ord(p2[translate-1]), 0, s[0]))
        f.set_start(t[0])
    f.add_arc(s[state-1], pn.Arc(0, 0, 0, s[0]))
    return f

def re_parse(expr):
    pattern = []
    _re_parse(expr, pattern)
    #f = re_to_fst(expr)
    ##print(draw(f))
    #draw(f).draw('./f.gv')
    return pattern

def _re_parse(expr, pattern):
    pattern.append("(")
    if (expr.decl().name() == 'str.to_re'):
        pattern.append( str(expr.arg(0)).strip(' " '))
    elif (expr.decl().name() == 're.*'):
        _re_parse(expr.arg(0), pattern)
    elif (expr.decl().name() == 're.+'):
        _re_parse(expr.arg(0), pattern)
        pattern.append("+")
    elif (expr.decl().name() == 're.++'):
        for i in range(expr.num_args()):
            _re_parse(expr.arg(i), pattern)
    elif (expr.decl().name() == 're.union'):
        _re_parse(expr.arg(0), pattern)
        for i in range(1, expr.num_args()):
            pattern.append("|")
            _re_parse(expr.arg(i), pattern)
    pattern.append(")")


def re_to_fst(expr):
    if (expr.decl().name() == 'str.to_re'):
        return pn.accep(str(expr.arg(0)).strip(' " ')).optimize()

    elif (expr.decl().name() == 're.*'):
        return re_to_fst(expr.arg(0)).closure().optimize()

    elif (expr.decl().name() == 're.+'):
        return re_to_fst(expr.arg(0)).closure_plus().optimize()

    elif (expr.decl().name() == 're.++'):
        return (re_to_fst(expr.arg(0)) + re_to_fst(expr.arg(1))).optimize()

    elif (expr.decl().name() == 're.union'):
        return pn.union(re_to_fst(expr.arg(0)), re_to_fst(expr.arg(1))).optimize()


def exprs_get_string_vars(exprs):
    var_lst = []
    for expr in exprs:
        for var in get_vars(expr):
            if z3.is_seq(var):
                var_lst.append(var)
    return var_lst

def transform_string(filename):
    c = z3.Context()
    s1 = z3.Solver(ctx=c)
    exprs = z3.parse_smt2_file(filename, ctx=c)
    exprs = [z3.simplify(expr) for expr in exprs]
    s1.add(exprs)
    #s1.set("timeout", 60)
    #t0 = time.time()
    #or_result = s1.check()
    #t1 = time.time()
    #or_time = t1-t0
    str_exprs, chs = extract(exprs)

    #print('origin result: ', or_result)
    #print('expressions:\n', exprs)
    #print('str_exprs:\n', str_exprs)

    '''
    sv = []
    for str_expr in str_exprs:
        if (z3.is_string_value(str_expr.arg(1))):
            clean_str = str(str_expr.arg(1)).strip(' " ')
            sv.append(clean_str)
        elif (z3.is_string_value(str_expr.arg(0))):
            clean_str = str(str_expr.arg(0)).strip(' " ')
            sv.append(clean_str)
    '''

    str_exprs_var_lst = exprs_get_string_vars(str_exprs)
    #print("chars: ", list(chs))
    #char_set.update(string.ascii_letters)
    #print("var_lst: ", str_exprs_var_lst)
    exprs_new = exprs
    if (len(str_exprs_var_lst) != 0):
        tv = []
        len_translate = random.randint(0, 5)
        len_scale = random.randint(1, 5)
        fst = gen_rand_len_fst_by_state(list(chs), translate = len_translate, scale = len_scale)
        #re_fst = []
        #print(fst)
        for str_expr in str_exprs:
            if (z3.is_string_value(str_expr.arg(0))):
                #print(list((str(str_expr.arg(0)).strip(' " ') @ fst).paths().ostrings()))
                tv.append(list((str(str_expr.arg(0)).strip(' " ') @ fst).paths().ostrings())[0])
            elif (z3.is_string_value(str_expr.arg(1))):
                #print(list((str(str_expr.arg(1)).strip(' " ') @ fst).paths().ostrings()))
                tv.append(list((str(str_expr.arg(1)).strip(' " ') @ fst).paths().ostrings())[0])
            elif (z3.is_re(str_expr.arg(1))):
                #print("isre: ", str_expr.arg(1))
                #print("pat: ", "".join(re_parse(str_expr.arg(1))))
                #re_fst.append(re_to_fst(str_expr.arg(1)))
                or_fst = re_to_fst(str_expr.arg(1))
                tr_fst = pn.compose(or_fst, fst).optimize().project("output")
                fst_to_z3 = parse_fst_str(str(draw(tr_fst)), c)
                #print("transfer re to z3: ", str_expr.arg(1), " to ", fst_to_z3)
                tv.append(fst_to_z3)

            elif (z3.is_re(str_expr.arg(0))):
                #print("isre: ", str_expr.arg(0))
                #print("pat: ", "".join(re_parse(str_expr.arg(0))))
                #re_fst.append(re_to_fst(str_expr.arg(0)))
                or_fst = re_to_fst(str_expr.arg(0))
                tr_fst = pn.compose(or_fst, fst).optimize().project("output")
                fst_to_z3 = parse_fst_str(str(draw(tr_fst)), c)
                #print("transfer re to z3: ", str_expr.arg(0), " to ", fst_to_z3)
                tv.append(fst_to_z3)
            else:
                print("nothing=========================")


        #print("tv", tv)
        #print("len check: ", len(str_exprs_var_lst) == len(str_exprs) and len(str_exprs) == len(tv))


        s_substitute = []
        for i in range(len(str_exprs_var_lst)):
            if(z3.is_re(tv[i]) and z3.is_re(str_exprs[i].arg(1))):
                s_substitute.append((str_exprs[i].arg(1), tv[i]))
            else:
                s_substitute.append((str_exprs[i], str_exprs_var_lst[i] == tv[i]))

        unique_var_lst = (list(set(str_exprs_var_lst)))
        #print("ulst: ", unique_var_lst)

        l_substitute = []
        for i in range(len(unique_var_lst)):
                l_substitute.append((z3.Length(unique_var_lst[i]), (z3.Length(unique_var_lst[i]) - len_translate) / len_scale))
        #print("l sub: ", l_substitute)
        exprs_new = [z3.substitute(expr, s_substitute) for expr in exprs]
        exprs_new = [z3.substitute(expr, l_substitute) for expr in exprs_new]
        #print(exprs_new)
    ar_vars = exprs_get_arith_vars(exprs_new)
    Q, Q_inv = uni_matrix_gen(len(ar_vars))
    r = rand_vector_gen(len(ar_vars))
    final_exprs = m_substitution(exprs_new, ar_vars, Q, r)
    #print("final_exprs", final_exprs)
    s1.reset()
    s1.add(final_exprs)
    #s1.set("timeout", 60)
    #t2 = time.time()
    #new_result = s1.check()
    #t3 = time.time()
    #new_time = t3-t2
    #print("new result: ", s1.check())
    #print("equivalent check: ", or_result == new_result)
    return s1.to_smt2()

def _is_re_expr(expr):
    return expr.decl().name() in ['str.in_re', '='] and (len(get_vars(expr)) == 1) and (_is_seq_const(expr.arg(0)) or _is_seq_const(expr.arg(0)))
def _is_seq_const(expr):
    return z3.is_seq(expr) and z3.is_const(expr)

def extract(exprs):
    re_exprs = []
    chs = set()
    for expr in exprs:
        _extract(expr, re_exprs, chs)
    return re_exprs, chs

def _extract(expr, re_res, chs):
    if z3.is_app(expr):
        #print(expr)
        #print(expr.num_args())
        #print(expr.decl().name() in ['str.in_re', '='])
        #print(z3.is_seq(expr.arg(0)))
        if(_is_re_expr(expr)):
            re_res.append(expr)

        if(z3.is_string_value(expr)):
            clstr = str(expr).strip(' " ')
            chs.update(set(clstr))

        for i in range(expr.num_args()):
            _extract(expr.arg(i), re_res, chs)

def parse_fst_str(fstr, c):
    enfa = EpsilonNFA()
    enfa.add_start_state(fstr.splitlines()[0].split()[0])
    for line in fstr.splitlines():
        args = line.split()
        if (len(args) == 4):
            enfa.add_transition(State(args[0]), Symbol(args[3]), State(args[1]))
        elif (len(args) == 1):
            enfa.add_final_state(State(args[0]))
        elif (len(args) == 3):
            enfa.add_transition(State(args[0]), Symbol(args[2]), State(args[1]))
        #print(args)
    reg = enfa.to_regex()
    #print("reg: ", reg)
    py_reg = re.compile(str(reg).replace('.', '').replace('$', '').replace('+', '|')).pattern

    #print("py_reg: ",py_reg)
    srep = sre_parse.parse(py_reg)
    #print("srep: ",srep.data[0])
    z3_re = regex_to_z3_expr(srep, c)
    #print(type(z3_re == z3.String('x')))
    return z3_re

def Minus(re1: z3.ReRef, re2: z3.ReRef) -> z3.ReRef:
    return z3.Intersect(re1, z3.Complement(re2))


def AnyChar(c) -> z3.ReRef:
    return z3.Range(chr(0), chr(127), ctx = c)
    # return z3.AllChar(z3.StringSort())


def category_regex(category: sre_constants._NamedIntConstant, c) -> z3.ReRef:
    if sre_constants.CATEGORY_DIGIT == category:
        return z3.Range("0", "9", ctx = c)
    elif sre_constants.CATEGORY_SPACE == category:
        return z3.Union(
            z3.Re(" ", ctx = c), z3.Re("\t", ctx = c), z3.Re("\n", ctx = c), z3.Re("\r", ctx = c), z3.Re("\f", ctx = c), z3.Re("\v", ctx = c)
        )
    elif sre_constants.CATEGORY_WORD == category:
        return z3.Union(
            z3.Range("a", "z", ctx = c), z3.Range("A", "Z", ctx = c), z3.Range("0", "9", ctx = c), z3.Re("_", ctx = c)
        )
    else:
        raise NotImplementedError(
            f"ERROR: regex category {category} not yet implemented"
        )


def regex_construct_to_z3_expr(regex_construct, c) -> z3.ReRef:
    node_type, node_value = regex_construct
    if sre_constants.LITERAL == node_type:  # a
        return z3.Re(chr(node_value), ctx = c)
    if sre_constants.NOT_LITERAL == node_type:  # [^a]
        return Minus(AnyChar(c), z3.Re(chr(node_value), ctx = c))
    if sre_constants.SUBPATTERN == node_type:
        _, _, _, value = node_value
        return regex_to_z3_expr(value, c)
    elif sre_constants.ANY == node_type:  # .
        return AnyChar(c)
    elif sre_constants.MAX_REPEAT == node_type:
        low, high, value = node_value
        if (0, 1) == (low, high):  # a?
            return z3.Option(regex_to_z3_expr(value, c))
        elif (0, sre_constants.MAXREPEAT) == (low, high):  # a*
            return z3.Star(regex_to_z3_expr(value, c))
        elif (1, sre_constants.MAXREPEAT) == (low, high):  # a+
            return z3.Plus(regex_to_z3_expr(value, c))
        else:  # a{3,5}, a{3}
            return z3.Loop(regex_to_z3_expr(value, c), low, high)
    elif sre_constants.IN == node_type:  # [abc]
        first_subnode_type, _ = node_value[0]
        if sre_constants.NEGATE == first_subnode_type:  # [^abc]
            return Minus(
                AnyChar(c),
                z3.Union(
                    [regex_construct_to_z3_expr(value, c) for value in node_value[1:]]
                ),
            )
        else:
            return z3.Union([regex_construct_to_z3_expr(value, c) for value in node_value])
    elif sre_constants.BRANCH == node_type:  # ab|cd
        _, value = node_value
        return z3.Union([regex_to_z3_expr(v, c) for v in value])
    elif sre_constants.RANGE == node_type:  # [a-z]
        low, high = node_value
        return z3.Range(chr(low), chr(high), ctx = c)
    elif sre_constants.CATEGORY == node_type:  # \d, \s, \w
        if sre_constants.CATEGORY_DIGIT == node_value:  # \d
            return category_regex(node_value, c)
        elif sre_constants.CATEGORY_NOT_DIGIT == node_value:  # \D
            return Minus(AnyChar(c), category_regex(sre_constants.CATEGORY_DIGIT, c))
        elif sre_constants.CATEGORY_SPACE == node_value:  # \s
            return category_regex(node_value, c)
        elif sre_constants.CATEGORY_NOT_SPACE == node_value:  # \S
            return Minus(AnyChar(c), category_regex(sre_constants.CATEGORY_SPACE, c))
        elif sre_constants.CATEGORY_WORD == node_value:  # \w
            return category_regex(node_value, c)
        elif sre_constants.CATEGORY_NOT_WORD == node_value:  # \W
            return Minus(AnyChar(c), category_regex(sre_constants.CATEGORY_WORD, c))
        else:
            raise NotImplementedError(
                f"ERROR: regex category {node_value} not implemented"
            )
    elif sre_constants.AT == node_type:
        if node_value in {
            sre_constants.AT_BEGINNING,
            sre_constants.AT_BEGINNING_STRING,
        }:  # ^a, \A
            raise NotImplementedError(
                f"ERROR: regex position {node_value} not implemented"
            )
        elif sre_constants.AT_BOUNDARY == node_value:  # \b
            raise NotImplementedError(
                f"ERROR: regex position {node_value} not implemented"
            )
        elif sre_constants.AT_NON_BOUNDARY == node_value:  # \B
            raise NotImplementedError(
                f"ERROR: regex position {node_value} not implemented"
            )
        elif node_value in {
            sre_constants.AT_END,
            sre_constants.AT_END_STRING,
        }:  # a$, \Z
            raise NotImplementedError(
                f"ERROR: regex position {node_value} not implemented"
            )
        else:
            raise NotImplementedError(
                f"ERROR: regex position {node_value} not implemented"
            )
    else:
        raise NotImplementedError(
            f"ERROR: regex construct {regex_construct} not implemented"
        )


def regex_to_z3_expr(regex: sre_parse.SubPattern, c) -> z3.ReRef:
    if 0 == len(regex.data):
        raise ValueError("ERROR: regex is empty")
    elif 1 == len(regex.data):
        return regex_construct_to_z3_expr(regex.data[0], c)
    else:
        return z3.Concat(
            [regex_construct_to_z3_expr(construct, c) for construct in regex.data]
        )

if __name__ == '__main__':
    sys.setrecursionlimit(10000000)
    if len(sys.argv) != 3:
        print('wrong argument, usage: python trans.py [input.smt2] [output.smt2]')
        sys.exit()
    #transform_smt2(sys.argv[1])
    try:
        #t_old, t_new = transform_string(sys.argv[1])
        f = open(sys.argv[1], 'r')
        f.close()    
    except:
        print("error, no such input file: ", sys.argv[1])
    
    try:
        f = open(sys.argv[2], 'w')
        print(transform_string(sys.argv[1]), file = f)
        f.close()    
    except Exception as e:
        print("error, transforming file: ", e)