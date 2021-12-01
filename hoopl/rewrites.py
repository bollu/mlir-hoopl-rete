#!/usr/bin/env python3
# describe the rewrites that we have, and code generate them.


class Variable:
    def __init__(self, name: str):
        self.name = name

def var(name): return Variable(name)

def op0(name):
    def op0_fn(lhs):
        return (lhs, name)
    return op0_fn

def op1(name):
    def op1_fn(lhs, rhs):
        return (lhs, name, rhs)
    return op1_fn

def op2(name):
    def op2_fn(lhs, rhs1, rhs2):
        return (lhs, name, rhs1, rhs2)
    return op2_fn


add = op2("add")
sub = op2("sub")
neg = op1("neg")
mul = op2("mul")
shr = op2("shr")

# sequence of instructions
class Program:                 
    def __init__(self, name):
        self.name = name
        self.insts = []
        pass

    def __matmul__(self, inst):
        self.insts.append(inst)

    def __str__(self):
        out = ""
        out += "***" + self.name + "***" + "\n"
        # out += "-" * len(self.name) + "\n"
        for inst in self.insts:
            out += str(inst[0]) + " = " + inst[1] +  "(" + ", ".join(map(str, inst[2:])) + ")"
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


add_to_shr = Rewrite("add_to_shr")
add_to_shr.src @ add("out", "x", "x")
add_to_shr.target @ shr("out", "x", 2)

print(add_to_shr)
