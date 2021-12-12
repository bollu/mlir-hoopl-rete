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
    elif kind == "set":
        env[lhs] = rhs[0]
    else:
        raise RuntimeError(f"unknonwn inst |{inst}|")
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
    print("query to Z3: |%s|" % (query))
    s = smt(query)
    is_eq = s.check() == sat


    # TODO: completely broken, need universal quantification over all inputs =)
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
    elif kind == "set":
        return 0
    else:
        raise RuntimeError(f"unknown instruction |{inst}|")

def cost_program(program):
    """Make sure cost is >= 1 so that when we take log, we get 0 as minimum"""
    return 1 + sum([cost_inst(inst) for inst in program.insts])

def rand_operand(regs: list):
   kind = random.choice(["reg", "constant"])
   if kind == "reg":
       return random.choice(regs)
   elif kind == "constant":
       const = random.randint(-4, -4)
       return const
   else:
       raise RuntimeError(f"unknown kind of random operand: |{kind}|")


# TODO: create a class to create a random instruction?
# python sucks, you can't randomly sample from a set! WTF!
def rand_inst(lhs: str, regs: list):
    """
    nregs: number of existing registers
    """
    # kind = random.choice(["add", "sub", "mul"])
    kind = random.choice(["add", "sub", "set"])
    if kind in ["add", "sub", "mul"]:
        rhs1 = rand_operand(regs)
        rhs2 = random.choice(regs) # this operand will always be a register
        inst = (lhs, kind, rhs1, rhs2)
        return inst 
    elif kind in "set":
        rhs1 = rand_operand(regs)
        inst = (lhs, kind, rhs1)
        return inst
    else:
        raise RuntimeError(f"unknown instruction kind |{kind}|")

GENSYM_GUID = 1000
def gensym_register():
    global GENSYM_GUID
    reg = f"%r{GENSYM_GUID}"
    GENSYM_GUID += 1
    return reg

def rand_program(name, ninputs, ninsts):
    insts = []
    regs = [f"reg{i}" for i in range(ninputs)]
    for _ in range(ninsts):
        lhs = gensym_register()
        # give a window of the last five registers when generating a random instruction
        # this makes sure that we actually use most registers
        REG_USE_WINDOW = 2
        inst = rand_inst(lhs, regs[-REG_USE_WINDOW:])
        insts.append(inst)
        regs.append(lhs)

    p =  Program(name, ninputs=ninputs, outreg=regs[-1], insts=insts)
    return prune_program(p)

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

# find instruction that defines a given register
def program_find_inst_defining_reg(p, reg):
    for (ix, inst) in enumerate(p.insts):
        if inst[0] == reg:
            return (ix, inst)
    return (-1, None)
        
def prune_program(p):
    """
    remove all instructions that are not backwards reachable from the return
    instruction.
    """
    allregs = set() # all registers found in backwards reachable search
    frontier = set([p.outreg]) # search frontier
    while True:
        newfrontier = set()
        for reg in frontier:
            allregs.add(reg)
            (_, inst) = program_find_inst_defining_reg(p, reg)
            # we have an input register on our hands
            if inst is None: continue
            assert inst is not None
            assert inst[0] == reg
            rhss = inst[2:]
            for rhs in rhss:
                if rhs in allregs: continue
                allregs.add(rhs)
                newfrontier.add(rhs)
        if not newfrontier: break
        frontier = newfrontier
    newinsts = []
    for inst in p.insts:
        if inst[0] in allregs:
            newinsts.append(inst)
    return Program(name=p.name, ninputs=p.ninputs, outreg=p.outreg, insts=newinsts)

def weighted_choice(xs, weights):
    return random.choices(xs, weights=weights, k=1)[0]

# Mutate a program.
def mutate_program(p):
    p = copy.deepcopy(p)
    vars_so_far = set()
    mutation_type = weighted_choice(["edit", "delete", "insert", "changeret"], weights=(5, 1, 5, 1))
    # print("p:\n--\n%s" % p)
    # print("mutation_type: %s" % mutation_type)
    # mutation_type = "delete"
    if mutation_type == "edit":
        if len(p.insts) == 0: return p # cannot edit empty program
        ix_to_edit = random.randint(0, len(p.insts)-1) # index: will insert at [location+eps, location+1)
        inst = p.insts[ix_to_edit]
        lhs = inst[0]
        regs = get_regs_program_upto_ix(p, ix_to_edit)
        inst = rand_inst(lhs, regs)
        p.insts[ix_to_edit] = inst
    elif mutation_type == "delete":
        if len(p.insts) == 0: return p # cannot delete empty program
        ix_to_del = random.randint(0, len(p.insts)-1)
        inst = p.insts[ix_to_del]; lhs = inst[0]
        REG_USE_WINDOW = 2
        regs = get_regs_program_upto_ix(p, ix_to_del)[:REG_USE_WINDOW]
        # regs = [rhs for rhs in inst[2:] if isinstance(rhs, str)]
        lhsnew  = random.choice(regs)
        del p.insts[ix_to_del]
        # replace all uses of LHS with a random reg
        for ix_inst in range(len(p.insts)):
            newinst = list(p.insts[ix_inst])
            for ix_rhs in range(2, len(newinst)):
                if newinst[ix_rhs] == lhs: newinst[ix_rhs] = lhsnew
            p.insts[ix_inst] = newinst

        if p.outreg == lhs: p.outreg = lhsnew

    elif mutation_type == "insert":
        # index: will insert such that q[<ix] = p[<ix] | q[ix] = new | q[>ix'] = p[ix'-1]
        ix_to_insert = random.randint(0, len(p.insts)) 
        regs = [f"reg{i}" for i in range(p.ninputs)]
        regs = get_regs_program_upto_ix(p, ix_to_insert)
        lhs = gensym_register()
        new_inst = rand_inst(lhs, regs)
        p.insts.append(new_inst) # add new instruction

        if random.randint(0, 1) == 1:
            p.outreg = lhs

    elif mutation_type == "changeret":
        p.outreg = random.choice(get_regs_program(prune_program(p)))
    else:
        raise RuntimeError("unknown mutation type|%s|" % (mutation_type))

    # Do not prune the program, since it makes it much harder for STOKE to get anything useful done!
    # pruned = prune_program(p)
    # print("q = p[%s]:\n--\n%s\nq.pruned:\n--\n%s" % (mutation_type, p, pruned))
    # input("pruned>")
    return p

def run_operand_concrete(operand, env):
    if isinstance(operand, int):
        return operand
    else:
        if operand not in env: 
            raise RuntimeError("unable to find operand |%s|" % (operand))
            # return None
        assert operand in env
        return env[operand]

def run_inst_concrete(inst, env):
    """
    run instruction.
    return True if failed to run
    """
    lhs = inst[0]
    kind = inst[1]
    rhss = [run_operand_concrete(operand, env) for operand in inst[2:]]
    # if None in rhss:
    #     return True # failed
    if kind == "add":
        env[lhs] = rhss[0] + rhss[1]
        pass
    elif kind == "sub":
        env[lhs] = rhss[0] - rhss[1]
    elif kind == "mul":
        env[lhs] = rhss[0] * rhss[1]
    elif kind == "set":
        env[lhs] = rhss[0]
    else:
        raise RuntimeError(f"unknown instruction |{inst}|")

def run_program_concrete(p, env):
    """
    Run program on input dictionary env: inx -> value
    """
    for i in range(p.ninputs):
        assert f"reg{i}" in env

    for inst in p.insts:
        failed = run_inst_concrete(inst, env)
        if failed: return None

    if p.outreg not in env: return None
    # assert p.outreg in env
    return env[p.outreg]

def run_stoke():
    ninputs = 2
    ninsts = 5
    p = rand_program("rand-0", ninputs, ninsts)
    # print("## STOKEing p### \n--------------\n%s" % p)
    # input("start STOKE[press any key to continue]>")
    q_best = p; score_best = 1;
    q = copy.deepcopy(p); score_q = 1

    N_TOTAL_STEPS = 2e4
    nsteps = 0
    while nsteps <= N_TOTAL_STEPS:
        nsteps += 1
        qnext = copy.deepcopy(q)

        N_CHAIN_STEPS = 20
        for _ in range(N_CHAIN_STEPS):
            qnext = mutate_program(qnext)
        qnext = prune_program(qnext)
        print("proposal[%s %%]:\n---------\n%s" % (100.0 * nsteps/N_TOTAL_STEPS, qnext))

        # PRUNE_THRESHOLD = 3
        # if len(qnext.insts) > PRUNE_THRESHOLD* len(p.insts):
        #     qnext = prune_program(qnext) # prune AFTER mutations

        assert p.ninputs == qnext.ninputs

        N_CONCRETE_RUNS = 3
        all_concrete_runs_matched = True

        score_qnext = 1
        for i in range(N_CONCRETE_RUNS):
            init_env = { f"reg{i}" :  random.randint(-3, 3) for i in range(p.ninputs) }
            if run_program_concrete(p, init_env) == run_program_concrete(qnext, init_env):
                score_qnext += 10
            else:
                all_concrete_runs_matched = False

        score_qnext -= cost_program(q) # weigh score down by cost
        score_qnext = math.exp(score_qnext)

        symbolic_equal = False
        if all_concrete_runs_matched:
            print("\trunning symbolic check...")
            symbolic_equal = symbolic_program_is_equiv(p, q)
            print("\t\t equal? %s" % symbolic_equal)
            if symbolic_equal:
                score_qnext += 100 # symbolic matching is very important


        # accept_threshold = log(score(q) / score(p))
        # rand > score_q / score_p <-> log rand > log(score_q) - log(score_p)
        if symbolic_equal or random.random() <= score_qnext/score_q:
            q = copy.deepcopy(qnext); score_q = score_qnext
            print("\taccepted proposal[score=%s]" % (score_q))

            if symbolic_equal and score_q > score_best:
                print("\taccepted as best^")
                score_best = score_q
                q_best = copy.deepcopy(q)

    return Rewrite("rewrite-0", p, q_best)

# random.seed(1)
# for i in range(10):
#     rewrite = run_stoke()
#     print("STOKEing\n---------\n")
#     if rewrite.src == rewrite.target:
#         print(rewrite.src)
#         print("\tno rewrite found")
#     else:
#         print(rewrite)
#         symbolic_equal = symbolic_program_is_equiv(rewrite.src, rewrite.target)
#         print("symbolic check equal? %s" % symbolic_equal)
#         cost_src = cost_program(rewrite.src)
#         cost_target = cost_program(rewrite.target)
#         percentage = 100.0 * (1.0 - cost_target/cost_src)
#         print("cost (original) %4.2f | cost(new) %4.2f | 1-tgt/src: %4.2f %%" % 
#                 (cost_src, cost_target, percentage))
#     input("result of stoke [press any key to continue]>")

random.seed(0)
ninsts = 100000
ninsts = 100
insts = []
regs = []
for _ in range(ninsts):
    lhs = gensym_register()
    # give a window of the last five registers when generating a random instruction
    # this makes sure that we actually use most registers
    REG_USE_WINDOW = 2
    kind = random.choice(["asm.add", "asm.int"])
    if regs and kind == "asm.add": # need instructions to add!
        # use last REG_USE_WINDOW registers.
        rhs1 = random.choice(regs[-REG_USE_WINDOW:])
        rhs2 = random.choice(regs[-REG_USE_WINDOW:])
        inst = (lhs, kind, rhs1, rhs2)
    else:
        rhs1 = random.randint(-4, 4)
        inst = (lhs, kind, rhs1)
    insts.append(inst)
    regs.append(lhs)

with open("rand-program-seed-0.mlir", "w") as f:
    f.write("func @main() {\n")
    for inst in insts:
        lhs = inst[0]; kind = inst[1]; rhss = inst[2:]
        rhss_str = ', '.join(map(str, rhss))
        f.write(f"  {lhs} = {kind} {rhss_str}\n")
    f.write("  return\n")
    f.write("}\n")
