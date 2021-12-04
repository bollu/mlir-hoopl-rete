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
    def __init__(self, name, insts=[]):
        self.name = name
        self.insts = insts
        pass

    def __matmul__(self, inst):
        self.insts.append(inst)

    def __str__(self):
        out = ""
        out += "***" + self.name + "***" + "\n"
        # out += "-" * len(self.name) + "\n"
        for inst in self.insts:
            out += str(inst[0]) + " = " + inst[1] +  "(" + ", ".join(map(str, inst[2:])) + ")" + "\n"
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

    if kind == "return":
        new_env = {}  # clear environment to keep only return value
        new_env[lhs] = rhs[0]
        return new_env

    if kind == "add":
        env[lhs] = rhs[0] + rhs[1]
    elif kind == "sub":
        env[lhs] = rhs[0] - rhs[1]
    elif kind == "mul":
        env[lhs] = rhs[0] * rhs[1]
    return env

def symbolic_program(p: Program):
    env = {}
    for inst in p.insts:
        env = symbolic_inst(env, inst)
    return env # return environment

# returns if two programs are symbolically equivalent
def symbolic_program_is_equiv(p: Program, q: Program):
    penv = symbolic_program(p); qenv = symbolic_program(q)
    outp = penv["out"]
    outq = qenv["out"]
    query = outp == outq

    max_input_ix = max(get_largest_input_ix_program(p), get_largest_input_ix_program(q))
    for i in range(max_input_ix):
        query = ForAll(Int(f"in{i}"), query)
    s = smt(query)
    is_eq = s.check() == sat


    # TODO: completely broken, need universal quantification over all inputs =)
    print(f"checking {outp == outq} : {is_eq}") 
    if is_eq: input(">")
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
    elif kind == "return":
        return 0 # return is zero cost
    else:
        raise RuntimeError(f"unknown instruction |{inst}|")

def cost_program(program):
    """Make sure cost is >= 1 so that when we take log, we get 0 as minimum"""
    return 1 + sum([cost_inst(inst) for inst in program.insts])

def rand_operand(totinputs: int):
   kind = random.choice(["new-input", "old-input", "constant"])
   if kind == "new-input":
       return (f"in{totinputs}", totinputs+1)
   elif kind == "old-input":
       ix = random.randint(0, totinputs-1)
       return (f"in{ix}", totinputs)
   elif kind == "constant":
       const = random.randint(-2, -2)
       return (const, totinputs)
   else:
       raise RuntimeError(f"unknown kind of random operand: |{kind}|")


# TODO: create a class to create a random instruction?
def rand_inst(totinputs: int):
    """
    totinputs: number of existing input variables
    """
    kind = random.choice(["add", "sub", "mul"])
    if kind in ["add", "sub", "mul"]:
        (lhs, totinputs) = (f"in{totinputs}", totinputs+1)
        (rhs1, totinputs) = rand_operand(totinputs)
        (rhs2, totinputs) = rand_operand(totinputs)
        inst = (lhs, kind, rhs1, rhs2)
        return (inst , totinputs)
    else:
        raise RuntimeError(f"unknown instruction kind |{kind}|")

def rand_program(name, ninsts):
    totinputs = 0
    insts = []
    for _ in range(ninsts):
        (inst, totinputs) = rand_inst(totinputs)
        insts.append(inst)

    ix = random.randint(0, totinputs-1); rhs = f"in{ix}"
    retinst = ("out", "return", rhs)
    insts.append(retinst)
    return Program(name, insts)

def get_largest_input_ix_inst(inst):
    """
    Get index (N) of the largest "inN" value used by the instruction.
    Returns 0 minimum
    """
    lhs = inst[0]
    name = inst[1]
    rhss = inst[2:]
    out = 0
    if lhs.startswith("in"):
        out = max(out, int(lhs.split("in")[1]))
    for rhs in rhss:
        if not isinstance(rhs, str): 
            assert isinstance(rhs, int) # is a constant arg.
            continue # continue
        assert isinstance(rhs, str)
        if rhs.startswith("in"):
            out = max(out, int(rhs.split("in")[1]))
    return out

def get_largest_input_ix_program(p):
    """
    Get index(N) of the largest "inN" value used by the program.
    Returns 0 minimum
    """
    out = 0
    for inst in p.insts:
        out = max(out, 1 + get_largest_input_ix_inst(inst))
    return out


def reindex_program_insts(p):
    """
    Reindex a program's instruction names to be in the set [0..|num_names_needed|]
    """

    nname = 0
    reindexed_names = {}
    q = Program(p.name, insts=[])
    for inst in p.insts:
        lhs = inst[0]
        kind = inst[1]
        rhss = list(inst[2:]) # convert to list for mutation


        # generate new name for rhs
        for i in range(len(rhss)):
            if rhss[i] not in reindexed_names:
                reindexed_names[rhss[i]]= f"in{nname}"
                nname += 1
            rhss[i] = reindexed_names[rhss[i]]

        # generate new name for lhs
        if lhs != "out":
            if lhs not in reindexed_names:
                reindexed_names[lhs] = f"in{nname}"
                nname += 1
            lhs = reindexed_names[lhs]

        q.insts.append(tuple([lhs, kind] + rhss))

    return q
        

# Mutate a program.
def mutate_program(p):
    p = copy.deepcopy(p)
    vars_so_far = set()
    mutation_type = random.choice(["edit", "delete", "insert"])
    # don't delete from program with only `return`.
    if len(p.insts) == 2:
        mutation_type = random.choice(["edit", "insert"])
    # print("mutation_type: %s" % mutation_type)
    if mutation_type == "edit":
        ix_to_edit = random.randint(0, len(p.insts)-2) # index: will insert at [location+eps, location+1)
        (inst, totinputs) = rand_inst(get_largest_input_ix_program(p) + 1)
        p.insts[ix_to_edit] = inst
        p = reindex_program_insts(p)
        return p
    elif mutation_type == "delete":
        # currently: does not delete the return instruction!
        ix = random.randint(0, len(p.insts)-2)
        del p.insts[ix]
        return p
    elif mutation_type == "insert":
        new_inst, totinputs = rand_inst(get_largest_input_ix_program(p) + 1)
        # TODO: allow changing the return instruction to be the new instruction
        ix_to_insert = random.randint(0, len(p.insts)) # index: will insert at [location+eps, location+1)
        if ix_to_insert == len(p.insts): # we are adding an instruction to the end
            del p.insts[-1] # delete return instruction
            p.insts.append(new_inst) # add new instruction
            p.insts.append(('out', 'return', new_inst[0])) # create new return instruction
        else: # not adding last instruction
            p.insts.insert(ix_to_insert, new_inst)
        p = reindex_program_insts(p)
        return p

def get_program_inputs(p):
    defs = set()
    inputs = set()
    for inst in p.insts:
        lhs = inst[0]
        kind = inst[1]
        rhss = inst[2:]
        for rhs in rhss:
            if not isinstance(rhs, str): continue
            if rhs not in defs: inputs.add(rhs)
        defs.add(lhs)
    return inputs


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
    elif kind == "return":
        env[lhs] = rhss[0]
    else:
        raise RuntimeError(f"unknown instruction |{inst}|")

def run_program_concrete(p, env):
    """
    Run program on input dictionary env: inx -> value
    """
    for inst in p.insts:
        run_inst_concrete(inst, env)
    return env["out"]

def run_stoke():
    ninsts = 4
    p = rand_program("rand-0", ninsts)
    p_inputs = get_program_inputs(p)
    log_score_p = - math.log(cost_program(p)) # score = log(1/cost)
    q_best = p; log_score_best = log_score_p
    q = p

    successful_mutations = 0
    while successful_mutations == 0:
        N_CHAIN_STEPS = 10
        for _ in range(N_CHAIN_STEPS):
            q = mutate_program(q)
        log_score_q = - math.log(cost_program(q)) # score = log(1/cost)

        q_inputs = get_program_inputs(q)
        # print(f"inputs: |{p_inputs}| == |{q_inputs}|")
        # TODO: this is really stupid, I should just fix the number of inputs in the Program.
        if p_inputs == q_inputs:
            N_CONCRETE_RUNS = 10
            all_concrete_runs_matched = True
            for i in range(N_CONCRETE_RUNS):
                init_env = { inp :  random.randint(-3, 3) for inp in p_inputs }
                if run_program_concrete(p, init_env) == run_program_concrete(q, init_env):
                    log_score_q += 1 # each correct matching output 
                else:
                    all_concrete_runs_matched = False

            print(f"p: {symbolic_program(p)['out']} | q: {symbolic_program(q)['out']}")
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
