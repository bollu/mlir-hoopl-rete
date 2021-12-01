#!/usr/bin/env python3
# describe the rewrites that we have, and code generate them.
from z3 import *
import random
import copy

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
    def __init__(self, name):
        self.name = name
        self.src = Program(self.name + ":source")
        self.target = Program(self.name + ":target")

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

def cost_program(program):
    return sum([cost_inst(inst) for inst in program.insts])

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


# Mutate a program.
def mutate_program(p):
    p = copy.deepcopy(p)
    vars_so_far = set()
    mutation_type = random.choice(["edit", "delete", "insert"])
    if mutation_type == "edit":
        ix_to_edit = random.randint(0, len(p.insts)) # index: will insert at [location+eps, location+1)
        p.insts[ix_to_edit] = rand_inst(get_largest_input_ix_program(p) + 1)
        return p
    elif mutation_type == "delete":
        ix = random.randint(0, len(p.insts)-1)
        del p.insts[ix]
        return p
    elif mutation_type == "insert":
        new_inst = rand_inst(get_largest_input_ix_program(p) + 1)
        ix_to_insert = random.randint(0, len(p.insts)) # index: will insert at [location+eps, location+1)
        p.insts.insert(ix_to_insert, new_inst)
        return p
