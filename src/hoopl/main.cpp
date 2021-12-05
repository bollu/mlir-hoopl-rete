#include <vector>
#include <list>
#include <iostream>
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch6/toyc.cpp

// #include "GRIN/GRINDialect.h"
// #include "Hask/HaskDialect.h"
// #include "Hask/HaskLazify.h"
// #include "Hask/HaskOps.h"
// #include "Interpreter.h"
// #include "Pointer/PointerDialect.h"
// #include "Runtime.h"
// #include "Unification/UnificationDialect.h"
// #include "Unification/UnificationOps.h"
// #include "lambdapure/Dialect.h"
// #include "lambdapure/Passes.h"

// #include "LZJIT/LZJIT.h"
// #include "RgnDialect.h"
// conversion
// https://github.com/llvm/llvm-project/blob/80d7ac3bc7c04975fd444e9f2806e4db224f2416/mlir/examples/toy/Ch6/toyc.cpp
// #include "mlir/Target/LLVMIR.h"
#include "mlir/Target/LLVMIR/Export.h"

// Execution
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include <llvm/ADT/ArrayRef.h>

extern "C" {
const char *__asan_default_options() { return "detect_leaks=0"; }
}

class AsmDialect : public mlir::Dialect {
public:
  explicit AsmDialect(mlir::MLIRContext *ctx);
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
  static llvm::StringRef getDialectNamespace() { return "asm"; }
};

class AsmType : public mlir::Type {
public:
  /// Inherit base constructors.
  using mlir::Type::Type;

  /// Support for PointerLikeTypeTraits.
  using mlir::Type::getAsOpaquePointer;
  static AsmType getFromOpaquePointer(const void *ptr) {
    return AsmType(static_cast<ImplType *>(const_cast<void *>(ptr)));
  }
  /// Support for isa/cast.
  static bool classof(Type type);
  AsmDialect &getDialect();
};

class AsmRegType
    : public mlir::Type::TypeBase<AsmRegType, AsmType, mlir::TypeStorage> {
public:
  using Base::Base;
  static AsmRegType get(mlir::MLIRContext *context) { return Base::get(context); }
};

class AsmIntOp :  public mlir::Op<AsmIntOp, mlir::OpTrait::OneResult, mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  static mlir::StringRef getOperationName() { return "asm.int"; };
  static mlir::ParseResult parse(mlir::OpAsmParser &parser, mlir::OperationState &result) { 
    mlir::IntegerAttr val;
    if(parser.parseAttribute<mlir::IntegerAttr>(val, "value", result.attributes)) {
      return mlir::failure();
    };
    result.addTypes(parser.getBuilder().getType<AsmRegType>());
    return mlir::success();
  };
  void print(mlir::OpAsmPrinter &p) { 
    p << getOperationName() << " " << this->getValue();
    // p.printGenericOp(this->getOperation());
  }
  int getValue() {
    mlir::IntegerAttr attr = this->getOperation()->getAttrOfType<mlir::IntegerAttr>("value");
    return attr.getInt();
  }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    int value) {
    state.addAttribute("value", builder.getI64IntegerAttr(value));
    state.addTypes(builder.getType<AsmRegType>());
  }
};


class AsmAddOp :  public mlir::Op<AsmAddOp, mlir::OpTrait::OneResult> {
public:
  using Op::Op;
  static mlir::StringRef getOperationName() { return "asm.add"; };

  mlir::Value lhs() { return this->getOperation()->getOperand(0); }
  mlir::Value rhs() { return this->getOperation()->getOperand(1); }
  static mlir::ParseResult parse(mlir::OpAsmParser &parser, mlir::OperationState &result) { 
    AsmRegType regty = parser.getBuilder().getType<AsmRegType>();
    mlir::OpAsmParser::OperandType lhs, rhs;
    if(parser.parseOperand(lhs) ||
        parser.parseComma() ||
        parser.parseOperand(rhs) ||
        parser.resolveOperand(lhs, regty, result.operands) ||
        parser.resolveOperand(rhs, regty, result.operands)) {
      return mlir::failure();
    }
    result.addTypes(parser.getBuilder().getType<AsmRegType>());
    return mlir::success();
  };
  void print(mlir::OpAsmPrinter &p) { 
    p << getOperationName() << " ";
    p.printOperand(lhs());
    p << ", ";
    p.printOperand(rhs());
  }
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Value lhs, mlir::Value rhs) {
    state.addOperands({lhs, rhs});
    state.addTypes(builder.getType<AsmRegType>());
  }
};
// === DIALECT ===
bool AsmType::classof(Type type) {
  return llvm::isa<AsmDialect>(type.getDialect());
}

AsmDialect &AsmType::getDialect() {
  return static_cast<AsmDialect &>(Type::getDialect());
}
AsmDialect::AsmDialect(mlir::MLIRContext *context)
    : Dialect(getDialectNamespace(), context, mlir::TypeID::get<AsmDialect>()) {
  // clang-format off
  // addOperations<IntToPtrOp, PtrToIntOp, PtrStringOp, FnToVoidPtrOp, PtrUndefOp>();
  // // addOperations<PtrToHaskValueOp> 
  // // addOperations<HaskValueToPtrOp>();
  // // addOperations<PtrBranchOp>(); 
  addOperations<AsmAddOp>();
  addOperations<AsmIntOp>();
  // addOperations<DoubleToPtrOp>();
  // addOperations<MemrefToVoidPtrOp>();
  // addOperations<PtrToFloatOp>();
  // addOperations<PtrGlobalOp>();
  // addOperations<PtrLoadGlobalOp>();
  // addOperations<PtrFnPtrOp>();
  // addOperations<PtrStoreGlobalOp>();
  // addOperations<PtrUnreachableOp>();
  // addOperations<PtrNotOp>();
  // addOperations<PtrLoadArrayOp>();
  addTypes<AsmRegType>();
  // clang-format on
}

mlir::Type AsmDialect::parseType(mlir::DialectAsmParser &parser) const {
  if (succeeded(parser.parseOptionalKeyword("reg"))) { // !ptr.void
    return AsmRegType::get(parser.getBuilder().getContext());
  }
  assert(false && "unable to parse reg type");
  return mlir::Type();
}
void AsmDialect::printType(mlir::Type type, mlir::DialectAsmPrinter &p) const {
  if (type.isa<AsmRegType>()) {
    p << "reg"; // !ptr.array
    return;
  }

  assert(false && "unknown type to print");
}

// == RETE IMPLEMENTATION ===
struct Token;
struct WME;
struct AlphaMemory;
struct JoinNode;
struct BetaMemory;
struct ProductionNode;

struct WME {
  static const int NFIELDS = 4;
  using FieldKindT = int;
  using FieldValueT = void*;
  FieldValueT fields[NFIELDS]; // lhs, kind, rhs0, rhs1

  FieldValueT get_field(WME::FieldKindT ty) const { 
    assert(ty >= 0 && ty < NFIELDS); return fields[(int)ty];
  }

};

std::ostream& operator << (std::ostream &os, WME w) {
    os << "(";
    for(int f = 0; f < WME::NFIELDS; ++f) {
      if (f > 0) { os << ", "; }
      os << w.fields[f];
    }
    os << ")";
    return os;
}

struct AlphaMemory {
  std::list<WME*> items; // every pointer must be valid
  std::list<JoinNode *> successors; // every pointer must be valid.
};

std::ostream &operator << (std::ostream &os, const AlphaMemory &am) {
    os <<  "(α-mem:" << am.items.size() << " ";
    for (WME *wme : am.items) os << *wme << " ";
    os << ")";
    return os;
}

struct ConstTestNode {
  WME::FieldKindT field_to_test;
  WME::FieldValueT field_must_equal;
  AlphaMemory *output_memory; // can be nullptr.
  std::vector<ConstTestNode *> children;

  // TODO: rethink this perhaps?
  static constexpr int FIELD_DUMMY = -1;

    ConstTestNode(WME::FieldKindT field_to_test, 
        WME::FieldValueT field_must_equal, AlphaMemory *output_memory) :
        field_to_test(field_to_test),
        field_must_equal(field_must_equal),
        output_memory(output_memory) {};


    static ConstTestNode *dummy_top() {
        ConstTestNode *node = new ConstTestNode(ConstTestNode::FIELD_DUMMY, (void*)0xDEADBEEF, nullptr);
        return node;
    }
};

std::ostream& operator << (std::ostream &os, const ConstTestNode &node) {
    if(node.field_to_test == ConstTestNode::FIELD_DUMMY) {
       return os << "(const-test dummy)";
    }
    return os << "(const-test " << node.field_to_test << " =? " << node.field_must_equal << ")";
}


// pg 22
struct Token {
    Token *parent; // items [0..i-1]
    int token_chain_ix; // index i of where this token is on the chain.
    WME *wme; // item i

    Token(WME *wme, Token *parent)  : parent(parent), wme(wme) {
        if (parent == nullptr) { token_chain_ix = 0; }
        else { token_chain_ix = parent->token_chain_ix+1; }
    }

    // implicitly stated on pages:
    // - pg 20
    // - pg 25 {With list-form tokens,the following statement is really a loop}
    // TODO: this is linear time! there _must_ be better lookup technology?
    WME *index(int ix) {
        if (! ((ix >= 0) && (ix< token_chain_ix))) {
          std::cerr << "ERROR: ix: " << ix << " token_chain_ix: " << token_chain_ix << " wme: " << *wme << "\n";
        }
        assert(ix >= 0);
        assert(ix <= token_chain_ix);
        if (ix == token_chain_ix) { return wme; }
        assert(parent != nullptr);
        return parent->index(ix);
    }
};

std::ostream &operator << (std::ostream &os, const Token &t) {
    os << "(";
    for(const Token *p = &t; p != nullptr; p = p->parent) {
        assert(p->wme);
        os << *(p->wme);
        if (p->parent != nullptr) { os << "->";}
    }
    os << ")";
    return os;
}

// pg 22
struct BetaMemory { 
    JoinNode *parent; // invariant: must be valid.
    std::list<Token*> items;
    std::vector<JoinNode *> children;

    // pg 23: dodgy! the types are different from BetaMemory and their children
    // updates
    virtual void join_activation(Token *t, WME *w);
};


std::ostream& operator <<(std::ostream &os, const BetaMemory &bm) {
    os << "(beta-memory ";
    for (Token *t: bm.items) {
        assert(t != nullptr);
        os << *t << " ";
    }
    os <<")";
    return os;
}

// pg 24
struct TestAtJoinNode {
    WME::FieldKindT field_of_arg1, field_of_arg2;
    int ix_in_token_of_arg2;

    bool operator == (const TestAtJoinNode &other) const {
        return field_of_arg1 == other.field_of_arg1 &&
            field_of_arg2 == other.field_of_arg2 &&
            ix_in_token_of_arg2 == other.ix_in_token_of_arg2;
    }
};

std::ostream& operator << (std::ostream &os, const TestAtJoinNode &test) {
    os << "(test-at-join ";
    os << test.field_of_arg1 << " ==  " << 
        test.ix_in_token_of_arg2 << "[" << test.field_of_arg2  << "]";
    os << ")";
    return os;
}


/// pg 24
struct JoinNode {
  AlphaMemory *amem_src; // invariant: must be valid
  BetaMemory *bmem_src; // can be nullptr

  std::vector<BetaMemory *> children;
  std::vector<TestAtJoinNode> tests;

  JoinNode() : amem_src(nullptr), bmem_src(nullptr) {};

   // pg 24
   void alpha_activation(WME *w) {
     assert(amem_src);
       if (bmem_src) {
         for (Token *t : bmem_src->items) {
           if (!this->perform_join_tests(t, w)) continue;
           for(BetaMemory *child: children) child->join_activation(t, w);
         }
       } else {
           for(BetaMemory *child: children) { child->join_activation(nullptr, w); }
       }
    }

   // pg 25
   void beta_activation(Token *t) {
     assert(this->amem_src);
     for(WME *w : amem_src->items) {
       if (!this->perform_join_tests(t, w)) continue;
       for(BetaMemory *child: children) child->join_activation(t, w);
     }
   }

    // pg 25
    bool perform_join_tests(Token *t, WME *w) const {
        if (!bmem_src) return true;
        assert(amem_src);

        for (TestAtJoinNode test : tests) {
          WME::FieldValueT arg1 = w->get_field(test.field_of_arg1);
          WME *wme2 = t->index(test.ix_in_token_of_arg2);
          WME::FieldValueT arg2 = wme2->get_field(test.field_of_arg2);
          if (arg1 != arg2) return false;
        }
        return true;
    }
};

std::ostream& operator << (std::ostream &os, const JoinNode &join) {
    os << "(join";
    for (TestAtJoinNode test : join.tests) {
        os << test;
    }
    os << ")";
    return os;
}

void BetaMemory::join_activation(Token *t, WME *w) {
        Token *new_token = new Token(w, t);
        items.push_front(new_token);
        for (JoinNode *child : children) { child->beta_activation(t); }
 }


// pg 37: inferred
struct ProductionNode : public BetaMemory {
  std::vector<Token *> items;
  using CallbackT = std::function<void(Token *t, WME *w)>;
  CallbackT callback;
  std::string rhs; // name of production

    void join_activation(Token *t, WME *w) override {
        t = new Token(w, t);
        items.push_back(t);
        assert(callback && "expected legal function pointer");
        callback(t, w);
        std::cout << "## (PROD " << *t <<  " ~ " << rhs << ") ##\n";
    }
};


std::ostream& operator << (std::ostream &os, const ProductionNode &production) {
    os << "(production " << production.rhs << ")";
    return os;
}



struct ReteContext {
  ConstTestNode *alpha_top;
  // alphabetically ordered for ease of use
  std::vector<AlphaMemory *> alphamemories;
  std::vector<BetaMemory *> betamemories;
  std::vector<ConstTestNode *> consttestnodes;
  std::vector<JoinNode *> joinnodes;
  std::vector<ProductionNode *> productions;

  // inferred from page 35: build_or_share_alpha memory:
  // { initialize am with any current WMEs }
  // presupposes knowledge of a collection of WMEs
  std::vector<WME*> working_memory;
};

// pg 21
void alpha_memory_activation(AlphaMemory *node, WME *w) {
    node->items.push_front(w);
    std::cerr << __PRETTY_FUNCTION__ << "| node: " << *node << " | wme: " << w << "\n";
    for (JoinNode *child : node->successors) child->alpha_activation(w);
}

// pg 15
// return whether test succeeded or not.
bool const_test_node_activation(ConstTestNode *node, WME *w) {
  std::cerr << __PRETTY_FUNCTION__ << "| node: " << *node << " | wme: " << w << "\n";

  // TODO: clean this up, this is a hack.
    if (node->field_to_test != -1) {
        if (w->get_field(node->field_to_test) != node->field_must_equal) {
            return false;
        }
    }

    if (node->output_memory) {
        alpha_memory_activation(node->output_memory, w);
    }
    for (ConstTestNode *c : node->children) {
        const_test_node_activation(c, w);
    }
    return true;
}


// pg 14
void addWME(ReteContext &r, WME *w) {
    r.working_memory.push_back(w);
    const_test_node_activation(r.alpha_top, w);
}

// pg 38
void update_new_node_with_matches_from_above(BetaMemory *beta) {
      JoinNode *join = beta->parent;
      std::vector<BetaMemory *> savedListOfChildren = join->children;
      // WTF?
      join->children = { beta };

      // push alpha memory through join node.
      for(WME *item : join->amem_src->items) { join->alpha_activation(item); }
      join->children = savedListOfChildren;
}


// pg 34
BetaMemory *build_or_share_beta_memory_node(ReteContext &r, JoinNode *parent) {
  // vv TODO: wut? thus looks ridiculous
  for (BetaMemory *child : parent->children) { return child; }

  BetaMemory *newbeta = new BetaMemory;
  r.betamemories.push_back(newbeta);
  newbeta->parent = parent;
  fprintf(stderr, "%s newBeta: %p | parent: %p\n", __FUNCTION__, newbeta, newbeta->parent);
  //newbeta->children = nullptr;
  //newbeta->items = nullptr;
  parent->children.push_back(newbeta);
  update_new_node_with_matches_from_above(newbeta);
  return newbeta;
}


// pg 34
JoinNode *build_or_share_join_node(ReteContext &r, BetaMemory *bmem, AlphaMemory *amem,
        std::vector<TestAtJoinNode> &tests) {
    // bmem can be nullptr in top node case.
    // assert(bmem != nullptr);
    assert(amem != nullptr);

    JoinNode *newjoin = new JoinNode;
    r.joinnodes.push_back(newjoin);
    newjoin->bmem_src = bmem;
    newjoin->tests = tests; newjoin->amem_src = amem;
    amem->successors.push_front(newjoin);
    if (bmem) { bmem->children.push_back(newjoin); }
    return newjoin;
}

// --- RETE FRONTEND ---

// inferred from discussion
enum FieldType {
    Const = 0,
    Var = 1
};

// inferred from discussion
struct Field {
    FieldType type;
    // TODO: review this dubious code.
    WME::FieldValueT v;

    static Field var(WME::FieldValueT name) {
        Field f; f.type = FieldType::Var; f.v = name; return f;
    }

    static Field constant(WME::FieldValueT name) {
        Field f; f.type = FieldType::Const; f.v = name; return f;
    }
};

// inferred from discussion
struct Condition {
    Field attrs[(int)WME::NFIELDS];
    Condition(Field ident, Field attr, Field val ) {
        attrs[0] = ident; attrs[1] = attr; attrs[2] = val;
    }
};


// implicitly defined on pg 35
void lookup_earlier_cond_with_field(const std::vector<Condition> &earlierConds, 
        WME::FieldValueT v, int *i, int *f2) {
    *i = earlierConds.size() - 1;
    *f2 = -1;

    for(auto it = earlierConds.rbegin(); it != earlierConds.rend(); ++it) {
        for (int j = 0; j < (int)WME::NFIELDS; ++j) {
            if (it->attrs[j].type != FieldType::Var) continue;
            if (it->attrs[j].v == v) { 
                *f2 = j;
                return;
            }
        }
        (*i)--;
    }
    *i = *f2 = -1;
}



// pg 35
// pg 35: supposedly, nearness is not a _hard_ requiement.
std::vector<TestAtJoinNode> get_join_tests_from_condition(ReteContext &_, Condition c, 
        const std::vector<Condition> &earlierConds) {
  std::vector<TestAtJoinNode> result;

    for(int f = 0; f < (int)WME::NFIELDS; ++f) {
        if (c.attrs[f].type != FieldType::Var) continue;
        // each occurence of variable v
        const WME::FieldValueT v = c.attrs[f].v;
        int i, f2;
        lookup_earlier_cond_with_field(earlierConds, v, &i, &f2);
        // nothing found
        if (i == -1)  { assert(f2 == -1); continue; }
        assert(i != -1); assert(f2 != -1);
        TestAtJoinNode test;
        test.field_of_arg1 = (WME::FieldKindT) f;
        test.ix_in_token_of_arg2 = i;
        test.field_of_arg2 = (WME::FieldKindT) f2;
        result.push_back(test);
    }
    return result;
};

// page 36
ConstTestNode *build_or_share_constant_test_node(ReteContext &r, 
        ConstTestNode *parent, 
        WME::FieldKindT f, WME::FieldValueT sym) {
    assert(parent != nullptr);
    // look for pre-existing node
    for (ConstTestNode *child: parent->children) {
        if (child->field_to_test == f && child->field_must_equal == sym) {
            return child;
        }
    }
    // build a new node
    ConstTestNode *newnode = new ConstTestNode(f, sym, nullptr);;
    r.consttestnodes.push_back(newnode);
    fprintf(stderr, "%s newconsttestnode: %p\n", __FUNCTION__, newnode);
    parent->children.push_back(newnode);
    // newnode->field_to_test = f; newnode->field_must_equal = sym;
    // newnode->output_memory = nullptr; 
    // newnode->children = nullptr;
    return newnode;
}

// implied in page 35: build_or_share_alpha_memory.
bool wme_passes_constant_tests(WME *w, Condition c) {
    for(int f = 0; f < (int)WME::NFIELDS; ++f) {
        if (c.attrs[f].type != FieldType::Const) continue;
        if (c.attrs[f].v != w->fields[f]) return false;
    }
    return true;
}

// pg 35: dataflow version
AlphaMemory *build_or_share_alpha_memory_dataflow(ReteContext &r, Condition c) {
    ConstTestNode *currentNode = r.alpha_top;
    for (int f = 0; f < (int)WME::NFIELDS; ++f) {
        if (c.attrs[f].type != FieldType::Const) continue;
        const WME::FieldValueT sym = c.attrs[f].v;
        currentNode = build_or_share_constant_test_node(r, currentNode, 
                (WME::FieldKindT)f, sym);
    }

    if (currentNode->output_memory != nullptr) {
        return currentNode->output_memory;
    } else {
      assert(currentNode->output_memory == nullptr);
      currentNode->output_memory = new AlphaMemory;
      r.alphamemories.push_back(currentNode->output_memory);
      // initialize AM with any current WMEs
      for (WME *w: r.working_memory) {
        // check if wme passes all constant tests
        if (wme_passes_constant_tests(w, c)) {
          alpha_memory_activation(currentNode->output_memory, w);
        }
      }
      return currentNode->output_memory;
    }
};

// page 36: hash version
AlphaMemory *build_or_share_alpha_memory_hashed(ReteContext &r, Condition c) {
    assert(false && "unimplemented");
};


// pg 37
// - inferred type of production node: 
ProductionNode *add_production(ReteContext &r, const std::vector<Condition> &lhs, 
    ProductionNode::CallbackT callback, std::string rhs) {
    // pseudocode: pg 33
    // M[1] <- dummy-top-node
    // build/share J[1] (a child of M[1]), the join node for c[1]
    // for i = 2 to k do
    //     build/share M[i] (a child of J[i-1]), a beta memory node
    //     build/share J[i] (a child of M[i]), the join node for ci
    // make P (a child of J[k]), the production node
  std::vector<Condition> earlierConds;

  std::vector<TestAtJoinNode> tests = 
      get_join_tests_from_condition(r, lhs[0], earlierConds);
    AlphaMemory *am = build_or_share_alpha_memory_dataflow(r, lhs[0]);

    BetaMemory *currentBeta = nullptr;
    JoinNode * currentJoin = build_or_share_join_node(r, currentBeta, am, tests);
    earlierConds.push_back(lhs[0]);

    // TDO: why not start with 0?
    for(int i = 1; i < (int)lhs.size(); ++i) {
        // get the current beat memory node M[i]
        currentBeta = build_or_share_beta_memory_node(r, currentJoin);
        // get the join node J[i] for condition c[u[
        tests = get_join_tests_from_condition(r, lhs[i], earlierConds);
        am = build_or_share_alpha_memory_dataflow(r, lhs[i]);
        currentJoin = build_or_share_join_node(r, currentBeta, am, tests);
        earlierConds.push_back(lhs[i]);
    }

    // build a new production node, make it a child of current node
    ProductionNode *prod = new ProductionNode; 
    r.productions.push_back(prod);
    prod->parent = currentJoin; // currentJoin is guaranteed to be valid
    fprintf(stderr, "%s prod: %p | parent: %p\n", __FUNCTION__, prod, prod->parent);
    prod->callback = callback;
    prod->rhs = rhs;
    currentJoin->children.push_back(prod);
    // update new-node-with-matches-from-above (the new production node)
    update_new_node_with_matches_from_above(prod);
    return prod;
}

// =========================
// END RETE, START EXAMPLES
// =========================





// === RETE OPTIMIZATION PASS ===
// === RETE OPTIMIZATION PASS ===
// === RETE OPTIMIZATION PASS ===
// === RETE OPTIMIZATION PASS ===
// === RETE OPTIMIZATION PASS ===


ReteContext *toRete(mlir::FuncOp f) {
  ReteContext *ctx = new ReteContext();
  assert (f.getBlocks().size() == 1 && "currently do not handle branching");

  const int ADD_OP_KIND = '+';
  const int INT_OP_KIND = 'i';

  for (mlir::Operation &op: f.getBlocks().front()) {
    if (AsmAddOp add = mlir::dyn_cast<AsmAddOp>(op)) {
      WME *wme = new WME;
      wme->fields[0] = add.getResult().getAsOpaquePointer();
      wme->fields[1] = (void *)ADD_OP_KIND;
      wme->fields[2] = add.lhs().getAsOpaquePointer();
      wme->fields[3] = add.rhs().getAsOpaquePointer();
      addWME(*ctx, wme);

    } else if (AsmIntOp i = mlir::dyn_cast<AsmIntOp>(op)) {
      WME *wme = new WME;
      wme->fields[0] = add.getResult().getAsOpaquePointer();
      wme->fields[1] = (void *)INT_OP_KIND;
      wme->fields[2] = (void *)i.getValue();
      wme->fields[3] = nullptr;
      addWME(*ctx, wme);
    } else if (mlir::ReturnOp ret = mlir::dyn_cast<mlir::ReturnOp>(op)) {
      // do nothing
    } else {
      llvm::errs() << op << "\n";
      assert(false && "unknown operation to RETE");
    }
  }
  assert(false && "unimplemented");
  return ctx;
};

mlir::FuncOp fromRete(mlir::MLIRContext *context, mlir::ModuleOp mod, ReteContext *rete) {
  assert(false && "unimplemented");
};

struct ReteOptimizationPass : public mlir::Pass {
  ReteOptimizationPass() : mlir::Pass(mlir::TypeID::get<ReteOptimizationPass>()){};
  mlir::StringRef getName() const override { return "ReteOptimization"; }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    auto newInst = std::make_unique<ReteOptimizationPass>(
        *static_cast<const ReteOptimizationPass *>(this));
    newInst->copyOptionValuesFrom(this);
    return newInst;
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = mlir::cast<mlir::ModuleOp>(this->getOperation());
    mlir::IRRewriter rewriter(mod.getContext());

    mod.walk([mod](mlir::FuncOp fn) {
        ReteContext *rete_ctx = toRete(fn);
        fn.erase();
        mlir::FuncOp newFn = fromRete(fn.getContext(), mod, rete_ctx);
        (void)newFn;
      });

  }
};


// === GREEDY PATTERN DRIVER PASS ===
// === GREEDY PATTERN DRIVER PASS ===
// === GREEDY PATTERN DRIVER PASS ===

class FoldAddPattern : public mlir::OpRewritePattern<AsmAddOp>  {

  public:
  FoldAddPattern(mlir::MLIRContext *context)
      : OpRewritePattern<AsmAddOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(AsmAddOp add,
                  mlir::PatternRewriter &rewriter) const override {
    AsmIntOp lhs = add.lhs().getDefiningOp<AsmIntOp>();
    AsmIntOp rhs = add.rhs().getDefiningOp<AsmIntOp>();
    if (!lhs || !rhs) { return mlir::failure(); }
    rewriter.replaceOpWithNewOp<AsmIntOp>(add, lhs.getValue() + rhs.getValue()); 
    return mlir::success();
  }
};

struct GreedyOptimizationPass : public mlir::Pass {
  GreedyOptimizationPass() : mlir::Pass(mlir::TypeID::get<GreedyOptimizationPass>()){};
  mlir::StringRef getName() const override { return "GreedyOptimizationPass"; }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    auto newInst = std::make_unique<GreedyOptimizationPass>(
        *static_cast<const GreedyOptimizationPass *>(this));
    newInst->copyOptionValuesFrom(this);
    return newInst;
  }

  void runOnOperation() override {
    mlir::OwningRewritePatternList patterns(&getContext());
    patterns.insert<FoldAddPattern>(&getContext());
    ::llvm::DebugFlag = true;
    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns)))) {
      llvm::errs() << "\n===greedy rewrite failed===\n";
      getOperation()->print(llvm::errs());
      llvm::errs() << "\n===\n";
      signalPassFailure();
      assert(false && "greedy rewrite failed");
    } else {
      // assert(false && "greedy rewrite succeeded");
      // success.
    }
    ::llvm::DebugFlag = false;
  }
};

// === PDL PASS ===
// === PDL PASS ===
// === PDL PASS ===


// === MAIN ===
// === MAIN ===
// === MAIN ===

using namespace llvm;
using namespace llvm::orc;

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  // mlir::registerInlinerPass();
  // mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerPass(
      "bench-greedy", "Rewrite using greedy pattern rewrite driver",
      []() -> std::unique_ptr<::mlir::Pass> { return std::make_unique<GreedyOptimizationPass>(); });

  mlir::registerPass(
      "bench-rete", "Rewrite using RETE algorithm",
      []() -> std::unique_ptr<::mlir::Pass> { return std::make_unique<ReteOptimizationPass>(); });

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<AsmDialect>();
  return failed(
      mlir::MlirOptMain(argc, argv, "Hoopl optimization drver", registry, true));
}

