#!/usr/bin/env python3
# describe the rewrites that we have, and code generate them.
# https://gist.github.com/bollu/65d85dd54524a0ea576a4ef4c87a36c0
from z3 import *
import random
import copy
import pudb

def smt(e): 
    """
    Create an SMT problem instance.
    Solve with s.check() == sat / s.check() == unsat
    Find model with model = s.model()
    """
    s = Solver()
    s.add([e])
    return s


def z3_to_py(v):
    if is_bool(v):
        return is_true(v)
    if is_int_value(v):
        return v.as_long()
    raise RuntimeError("unknown z3 value to be coerced |%s|" % (v, ))


def op0(kind):
    def op0_fn(lhs):
        return (lhs, kind)
    return op0_fn

def op1(kind):
    def op1_fn(lhs, rhs):
        return (lhs, kind, rhs)
    return op1_fn

def op2(kind):
    def op2_fn(lhs, rhs1, rhs2):
        return (lhs, kind, rhs1, rhs2)
    return op2_fn


add = op2("add")
sub = op2("sub")
neg = op1("neg")
mul = op2("mul")
shr = op2("shr")


# sequence of instructions
# TODO: change representation to have 'out value'
class Program:                 
    def __init__(self, name, ninputs=0, outreg=None, insts=[]):
        self.name = name
        self.ninputs = ninputs
        self.outreg = outreg
        self.insts = insts
        pass

    def __matmul__(self, inst):
        self.insts.append(inst)

    def __str__(self):
        out = ""
        out += "***" + self.name + f"(ninputs={self.ninputs})" + "***" + "\n"
        # out += "-" * len(self.name) + "\n"
        for inst in self.insts:
            out += str(inst[0]) + " = " + inst[1] +  "(" + ", ".join(map(str, inst[2:])) + ")" + "\n"
        out += f"return {self.outreg}\n"
        return out
    __repr__ = __str__

class Rewrite:
    def __init__(self, name, src=None, target=None):
        self.name = name
        if src is None:
            self.src = Program(self.name + ":source")
        else:
            assert isinstance(src, Program)
            self.src = src
        
        if target is None:
            self.target = Program(self.name + ":target")
        else:
            assert isinstance(target, Program)
            self.target = target

    def __str__(self):
        out = ""
        out += "v" * len(self.name) + "\n"
        out += self.name + "\n\n"
        out += str(self.src) + "\n\n"
        out += str(self.target) + "\n"
        out += "^" * len(self.name) + "\n"
        return out

    __repr__ = __str__


# https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp
# https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/InstCombine/InstCombineAndOrXor.cpp

def symbolic_operand(env, rand):
    if isinstance(rand, int):
        return rand
    else:
        assert isinstance(rand, str)
        if rand in env:
            return env[rand]
        else:
            env[rand] = Int(rand)
            return env[rand]

def symbolic_inst(env, inst):
    # (lhs, name, rhss)
    lhs = inst[0]
    kind = inst[1]
    rhs = [symbolic_operand(env, rand) for rand in inst[2:]]


    if kind == "add":
        env[lhs] = rhs[0] + rhs[1]
    elif kind == "sub":
        env[lhs] = rhs[0] - rhs[1]
    elif kind == "mul":
        env[lhs] = rhs[0] * rhs[1]
    return env

def symbolic_program(p: Program):
    env = { f"reg{i}": Int(f"reg{i}") for i in range(p.ninputs) }
    for inst in p.insts:
        env = symbolic_inst(env, inst)
    return env[p.outreg] # return environment

# returns if two programs are symbolically equivalent
def symbolic_program_is_equiv(p: Program, q: Program):
    outp = symbolic_program(p); outq = symbolic_program(q)
    query = outp == outq

    if p.ninputs != q.ninputs: return False

    for i in range(p.ninputs):
        query = ForAll(Int(f"reg{i}"), query)
    s = smt(query)
    is_eq = s.check() == sat


    # TODO: completely broken, need universal quantification over all inputs =)
    print(f"checking {query} : {is_eq}") 
    return is_eq

def cost_inst(inst):
    lhs = inst[0]
    kind = inst[1]
    if kind == "add":
        return 1
    elif kind == "sub":
        return 1
    elif kind == "mul":
        return 4
    else:
        raise RuntimeError(f"unknown instruction |{inst}|")

def cost_program(program):
    """Make sure cost is >= 1 so that when we take log, we get 0 as minimum"""
    return 1 + sum([cost_inst(inst) for inst in program.insts])

def rand_operand(regs: list):
   kind = random.choice(["reg", "constant"])
   if kind == "reg":
       return (random.choice(regs), regs)
   elif kind == "constant":
       const = random.randint(-2, -2)
       return (const, regs)
   else:
       raise RuntimeError(f"unknown kind of random operand: |{kind}|")


# TODO: create a class to create a random instruction?
# python sucks, you can't randomly sample from a set! WTF!
def rand_inst(lhs: str, regs: list):
    """
    nregs: number of existing registers
    """
    kind = random.choice(["add", "sub", "mul"])
    if kind in ["add", "sub", "mul"]:
        (rhs1, nregs) = rand_operand(regs)
        (rhs2, nregs) = rand_operand(regs)
        inst = (lhs, kind, rhs1, rhs2)
        return (inst , nregs)
    else:
        raise RuntimeError(f"unknown instruction kind |{kind}|")

GENSYM_GUID = 0
def gensym_register():
    global GENSYM_GUID
    reg = f"reg{GENSYM_GUID}"
    GENSYM_GUID += 1
    return reg

def rand_program(name, ninputs, ninsts):
    insts = []
    regs = [f"reg{i}" for i in range(ninputs)]
    for _ in range(ninsts):
        lhs = gensym_register()
        (inst, nregs) = rand_inst(lhs, regs)
        insts.append(inst)
        regs.append(lhs)

    return Program(name, ninputs=ninputs, outreg=random.choice(regs), insts=insts)

def get_regs_inst(inst):
    """
    Get index (the value N) of the largest "regN" value used by the instruction.
    Returns -1 minimum
    """
    lhs = inst[0]
    name = inst[1]
    regs = [lhs]
    for rhs in inst[2:]:
        if isinstance(rhs, str):
            assert rhs.startswith("reg")
            regs.append(rhs)
    return regs


# get registers in program of insts[:ix]
def get_regs_program_upto_ix(p, ix):
    regs = [f"reg{i}" for i in range(p.ninputs)]
    for inst in p.insts[:ix]:
        regs.extend(get_regs_inst(inst))
    return list(set(regs))

def get_regs_program(p):
    """
    Get index(the value N) of the largest "regN" value used by the program.
    Returns 0 minimum
    """
    return get_regs_program_upto_ix(p, len(p.insts))


def is_reg_used(p:Program, reg:str):
    if p.outreg == reg: return True
    for inst in p.insts:
        lhs = inst[0]
        rhss = inst[2:]
        if lhs == reg: return True
        if reg in rhss: return True
    return False
        

# Mutate a program.
def mutate_program(p):
    p = copy.deepcopy(p)
    vars_so_far = set()
    mutation_type = random.choice(["edit", "delete", "insert", "changeret"])
    # print("mutation_type: %s" % mutation_type)
    if mutation_type == "edit":
        ix_to_edit = random.randint(0, len(p.insts)-1) # index: will insert at [location+eps, location+1)
        inst = p.insts[ix_to_edit]
        lhs = inst[0]
        regs = get_regs_program_upto_ix(p, ix_to_edit)
        (inst, _) = rand_inst(lhs, regs)
        p.insts[ix_to_edit] = inst
        return p
    elif mutation_type == "delete":
        if len(p.insts) == 1: return p # cannot delete empty program
        ix = random.randint(0, len(p.insts)-1)
        inst = p.insts[ix]; lhs = inst[0]
        if is_reg_used(p, lhs): return p # cannot delete register in use
        del p.insts[ix]
        return p
    elif mutation_type == "insert":
        # index: will insert such that q[<ix] = p[<ix] | q[ix] = new | q[>ix'] = p[ix'-1]
        ix_to_insert = random.randint(0, len(p.insts)) 
        regs = [f"reg{i}" for i in range(p.ninputs)]
        regs = get_regs_program_upto_ix(p, ix_to_insert)
        lhs = gensym_register()
        new_inst, _ = rand_inst(lhs, regs)
        p.insts.append(new_inst) # add new instruction
        return p
    elif mutation_type == "changeret":
        p.outreg = random.choice(get_regs_program(p))
        return p


def run_operand_concrete(operand, env):
    if isinstance(operand, int):
        return operand
    else:
        assert operand in env
        return env[operand]

def run_inst_concrete(inst, env):
    lhs = inst[0]
    kind = inst[1]
    rhss = [run_operand_concrete(operand, env) for operand in inst[2:]]
    if kind == "add":
        env[lhs] = rhss[0] + rhss[1]
        pass
    elif kind == "sub":
        env[lhs] = rhss[0] - rhss[1]
    elif kind == "mul":
        env[lhs] = rhss[0] * rhss[1]
    else:
        raise RuntimeError(f"unknown instruction |{inst}|")

def run_program_concrete(p, env):
    """
    Run program on input dictionary env: inx -> value
    """
    for i in range(p.ninputs):
        assert f"reg{i}" in env

    for inst in p.insts:
        run_inst_concrete(inst, env)
    assert p.outreg in env
    return env[p.outreg]

def run_stoke():
    ninsts = 4
    ninputs = 2
    p = rand_program("rand-0", ninputs, ninsts)
    log_score_p = - math.log(cost_program(p)) # score = log(1/cost)
    q_best = p; log_score_best = log_score_p
    q = p

    successful_mutations = 0
    while successful_mutations == 0:
        N_CHAIN_STEPS = 10
        for _ in range(N_CHAIN_STEPS):
            q = mutate_program(q)
        log_score_q = - math.log(cost_program(q)) # score = log(1/cost)

        assert p.ninputs == q.ninputs

        N_CONCRETE_RUNS = 10
        all_concrete_runs_matched = True
        for i in range(N_CONCRETE_RUNS):
            init_env = { f"reg{i}" :  random.randint(-3, 3) for i in range(p.ninputs) }
            if run_program_concrete(p, init_env) == run_program_concrete(q, init_env):
                log_score_q += 1 # each correct matching output 
            else:
                all_concrete_runs_matched = False

        # run symbolic equivalence if concrete runs all match.
        if all_concrete_runs_matched and symbolic_program_is_equiv(p, q):
            log_score_q += 10 # correctness is extremely important.

            if log_score_q >= log_score_best:
                log_score_best = log_score_q
                q_best = q
                successful_mutations += 1

        # accept_threshold = log(score(q) / score(p))
        # rand > score_q / score_p <-> log rand > log(score_q) - log(score_p)
        if math.log(random.random()) > log_score_q - log_score_p:
            q = p
    return Rewrite("rewrite-0", p, q_best)

random.seed(0)
for i in range(10):
    print(run_stoke())
