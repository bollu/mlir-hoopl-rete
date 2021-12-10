#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <list>
#include <vector>

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
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
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
  static AsmRegType get(mlir::MLIRContext *context) {
    return Base::get(context);
  }
};

class AsmIntOp : public mlir::Op<AsmIntOp, mlir::OpTrait::OneResult,
                                 mlir::OpTrait::ZeroOperands> {
public:
  using Op::Op;
  static mlir::StringRef getOperationName() { return "asm.int"; };
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
    mlir::IntegerAttr val;
    if (parser.parseAttribute<mlir::IntegerAttr>(val, "value",
                                                 result.attributes)) {
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
    mlir::IntegerAttr attr =
        this->getOperation()->getAttrOfType<mlir::IntegerAttr>("value");
    return attr.getInt();
  }

  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    int value) {
    state.addAttribute("value", builder.getI64IntegerAttr(value));
    state.addTypes(builder.getType<AsmRegType>());
  }
};

class AsmAddOp : public mlir::Op<AsmAddOp, mlir::OpTrait::OneResult> {
public:
  using Op::Op;
  static mlir::StringRef getOperationName() { return "asm.add"; };

  mlir::Value lhs() { return this->getOperation()->getOperand(0); }
  mlir::Value rhs() { return this->getOperation()->getOperand(1); }
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
    AsmRegType regty = parser.getBuilder().getType<AsmRegType>();
    mlir::OpAsmParser::OperandType lhs, rhs;
    if (parser.parseOperand(lhs) || parser.parseComma() ||
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
// == RETE IMPLEMENTATION ===
// == RETE IMPLEMENTATION ===
// == RETE IMPLEMENTATION ===
// == RETE IMPLEMENTATION ===
// == RETE IMPLEMENTATION ===
// == RETE IMPLEMENTATION ===
// TODO: 2.5.1: On the implementation of lists
// TODO 2.6 (I don't actually need this): adding/removing productions
// TODO: 2.7: Negated conditions.
//
//
// β ~ [Token]              α ~ [WME]
//   \                       /
//    \                     /
//     \                   /
//      \                 /
// Left  \               / Right
//        \             /
//         \           /
//          \         /
//           \       /
//            \     /
//            Join node
struct Token;
struct WME;
struct AlphaWMEsMemory;
struct JoinNode;
struct BetaTokensMemory;
struct ProductionNode;
struct NegativeJoinResult;
struct NegativeNode;

// random chunk of memory.
// pg 21
struct WME {
  static const int NFIELDS = 4;
  using FieldKindT = int;
  using FieldValueT = int64_t;
  FieldValueT fields[NFIELDS]; // lhs, kind, rhs0, rhs1
  // void *data_instructionPtr;

  FieldValueT get_field(WME::FieldKindT ty) const {
    assert(ty >= 0 && ty < NFIELDS);
    return fields[(int)ty];
  }

  std::list<AlphaWMEsMemory *> parentAlphas; // α mems that contain this WME
  std::list<Token *> parentTokens; // tokens such that token->wme == this wme.
  std::list<NegativeJoinResult *> negativeJoinResults; // list of negative join results involving this WME.

  // pg 30
  void remove();
};

std::ostream &operator<<(std::ostream &os, WME w) {
  os << "(";
  for (int f = 0; f < WME::NFIELDS; ++f) {
    if (f > 0) {
      os << ", ";
    }
    os << w.fields[f];
  }
  os << ")";
  return os;
}

// α mem ---> successors :: [join node]
// pg 21
struct AlphaWMEsMemory {
  std::list<WME *> items;           // every pointer must be valid
  std::list<JoinNode *> successors; // every pointer must be valid.
};

std::ostream &operator<<(std::ostream &os, const AlphaWMEsMemory &am) {
  os << "(α-mem:" << am.items.size() << " ";
  for (WME *wme : am.items)
    os << *wme << " ";
  os << ")";
  return os;
}

// pg 14
struct ConstTestNode {
  WME::FieldKindT field_to_test;
  WME::FieldValueT field_must_equal;
  AlphaWMEsMemory *output_memory; // can be nullptr.
  std::vector<ConstTestNode *> successors;

  // TODO: rethink this perhaps?
  static constexpr int FIELD_DUMMY = -1;

  ConstTestNode(WME::FieldKindT field_to_test,
                WME::FieldValueT field_must_equal,
                AlphaWMEsMemory *output_memory)
      : field_to_test(field_to_test), field_must_equal(field_must_equal),
        output_memory(output_memory){};

  static ConstTestNode *dummy_top() {
    ConstTestNode *node = new ConstTestNode(ConstTestNode::FIELD_DUMMY,
                                            int64_t(-42), nullptr);
    return node;
  }
};

std::ostream &operator<<(std::ostream &os, const ConstTestNode &node) {
  if (node.field_to_test == ConstTestNode::FIELD_DUMMY) {
    return os << "(const-test dummy)";
  }
  return os << "(const-test " << node.field_to_test << " =? "
            << node.field_must_equal << ")";
}

// pg 22
// pg 41: negative nodes
struct Token {
  Token *parentToken;           // items [0..i-1]
  int token_chain_ix;           // index i of where this token is on the chain.
  WME *wme;                     // item i
  BetaTokensMemory *parentBeta; // beta node where this token lives.
  std::list<Token *> children;  // list of tokens whose parent is this token.
  std::list<NegativeJoinResult *> joinResults; // used for tokens in -ve nodes.

  // page 30
  Token(BetaTokensMemory *parentBeta, WME *wme, Token *parentToken)
      : parentToken(parentToken), wme(wme), parentBeta(parentBeta) {
    if (parentToken == nullptr) {
      token_chain_ix = 0;
    } else {
      token_chain_ix = parentToken->token_chain_ix + 1;
    }
  }

  // implicitly stated on pages:
  // - pg 20
  // - pg 25 {With list-form tokens,the following statement is really a loop}
  // TODO: this is linear time! there _must_ be better lookup technology?
  WME *index(int ix) {
    // what is this <= versus < nonsense?
    if (!((ix >= 0) && (ix <= token_chain_ix))) {
      std::cerr << "ERROR: ix: " << ix << " token_chain_ix: " << token_chain_ix
                << " wme: " << *wme << "\n";
    }
    assert(ix >= 0);
    assert(ix <= token_chain_ix);
    if (ix == token_chain_ix) {
      return wme;
    }
    assert(parentToken != nullptr);
    return parentToken->index(ix);
  }

  // pg 31
  static void delete_token_and_descendants(Token *t);
};

std::ostream &operator<<(std::ostream &os, const Token &t) {
  os << "(";
  for (const Token *p = &t; p != nullptr; p = p->parentToken) {
    assert(p->wme);
    os << *(p->wme);
    if (p->parentToken != nullptr) {
      os << "->";
    }
  }
  os << ")";
  return os;
}

// pg 22
// parent :: join node ---> β memory ---> successors :: [join node]
struct BetaTokensMemory {
  JoinNode *parent; // invariant: must be valid.
  std::list<Token *> items;
  std::vector<JoinNode *> successors;

  // pg 23: dodgy! the types are different from BetaTokensMemory and their
  // successors updates pg 30: revised from pg 23 (rete calls this left
  // activation).
  virtual void join_activation(Token *t, WME *w);
};

std::ostream &operator<<(std::ostream &os, const BetaTokensMemory &bm) {
  os << "(beta-memory ";
  for (Token *t : bm.items) {
    assert(t != nullptr);
    os << *t << " ";
  }
  os << ")";
  return os;
}

// pg 24
struct TestAtJoinNode {
  WME::FieldKindT field_of_arg1, field_of_arg2;
  int ix_in_token_of_arg2;

  bool operator==(const TestAtJoinNode &other) const {
    return field_of_arg1 == other.field_of_arg1 &&
           field_of_arg2 == other.field_of_arg2 &&
           ix_in_token_of_arg2 == other.ix_in_token_of_arg2;
  }
};

std::ostream &operator<<(std::ostream &os, const TestAtJoinNode &test) {
  os << "(test-at-join ";
  os << test.field_of_arg1 << " ==  " << test.ix_in_token_of_arg2 << "["
     << test.field_of_arg2 << "]";
  os << ")";
  return os;
}

// α -\
//     *----- join --> successors :: [β]
//     /    extend/filter β with α
// β -/
/// pg 24
struct JoinNode {
  AlphaWMEsMemory *amem_src;  // invariant: must be valid
  BetaTokensMemory *bmem_src; // can be nullptr

  std::vector<BetaTokensMemory *> successors;
  std::vector<TestAtJoinNode> tests;

  JoinNode() : amem_src(nullptr), bmem_src(nullptr){};

  // pg 24
  void alpha_activation(WME *w) {
    assert(w);
    assert(amem_src);
    if (bmem_src) {
      for (Token *t : bmem_src->items) {
        if (!this->perform_join_tests(t, w))
          continue;
        for (BetaTokensMemory *succ : successors)
          succ->join_activation(t, w);
      }
    } else {
      // wut? it can be null?
      // why is it null?
      for (BetaTokensMemory *succ : successors) {
        succ->join_activation(nullptr, w);
      }
    }
  }

  // pg 25
  void join_activation(Token *t) {
    assert(t && "token pointer invalid!");
    assert(this->amem_src);
    for (WME *w : amem_src->items) {
      if (!this->perform_join_tests(t, w))
        continue;
      for (BetaTokensMemory *succ : successors) {
        succ->join_activation(t, w);
      }
    }
  }

  // pg 25
  bool perform_join_tests(Token *t, WME *w) const {
    assert(w);
    assert(t && "token pointer invalid!");
    if (!bmem_src)
      return true;
    assert(amem_src);

    for (TestAtJoinNode test : tests) {
      WME::FieldValueT arg1 = w->get_field(test.field_of_arg1);
      std::cerr << "t: [" << t << "]\n";
      std::cerr << "test: [" << test.ix_in_token_of_arg2 << "]"
                << "\n";
      WME *wme2 = t->index(test.ix_in_token_of_arg2);
      WME::FieldValueT arg2 = wme2->get_field(test.field_of_arg2);
      if (arg1 != arg2)
        return false;
    }
    return true;
  }
};

std::ostream &operator<<(std::ostream &os, const JoinNode &join) {
  os << "(join";
  for (TestAtJoinNode test : join.tests) {
    os << test;
  }
  os << ")";
  return os;
}

// pg 23, pg 30: revised from pg 23
void BetaTokensMemory::join_activation(Token *t, WME *w) {
  assert(w);
  Token *new_token = new Token(this, w, t);
  items.push_front(new_token);
  for (JoinNode *succ : successors) {
    succ->join_activation(new_token);
  }
}

// pg 37: inferred
struct ProductionNode : public BetaTokensMemory {
  std::vector<Token *> items;
  using CallbackT = std::function<void(Token *t, WME *w)>;
  CallbackT callback;
  std::string rhs; // name of production

  void join_activation(Token *t, WME *w) override {
    t = new Token(this, w, t);
    items.push_back(t);
    assert(callback && "expected legal function pointer");
    callback(t, w);
    std::cout << "## (PROD " << *t << " ~ " << rhs << ") ##\n";
  }
};

std::ostream &operator<<(std::ostream &os, const ProductionNode &production) {
  os << "(production " << production.rhs << ")";
  return os;
}

// ---------RETE deletion support--------
// ---------RETE deletion support--------
// ---------RETE deletion support--------
// ---------RETE deletion support--------

// pg 30
void WME::remove() {
  for (AlphaWMEsMemory *alpha : this->parentAlphas) {
    alpha->items.remove(this);
  }
  for (Token *t : this->parentTokens) {
    Token::delete_token_and_descendants(t);
  }
  this->parentTokens.clear();
}

// pg 31
void Token::delete_token_and_descendants(Token *t) {
  for (Token *child : t->children) {
    Token::delete_token_and_descendants(child);
  }
  t->children.clear();
  t->parentBeta->items.remove(t);
  t->wme->parentTokens.remove(t);
  t->parentToken->children.remove(t);
  delete (t);
}

struct ReteContext {
  ConstTestNode *alpha_top;
  // alphabetically ordered for ease of use
  std::vector<AlphaWMEsMemory *> alphamemories;
  std::vector<BetaTokensMemory *> betamemories;
  std::vector<ConstTestNode *> consttestnodes;
  std::vector<JoinNode *> joinnodes;
  std::vector<ProductionNode *> productions;

  // inferred from page 35: build_or_share_alpha memory:
  // { initialize am with any current WMEs }
  // presupposes knowledge of a collection of WMEs
  std::list<WME *> working_memory;

  ReteContext() {
    this->alpha_top = ConstTestNode::dummy_top();
    this->consttestnodes.push_back(this->alpha_top);
  }

  int64_t gensym_counter = 1;

  int64_t gensym() {
    return gensym_counter++;
  }

};

// pg 21
// revised for deletion: pg 30
void alpha_memory_activation(AlphaWMEsMemory *node, WME *w) {
  node->items.push_front(w);
  w->parentAlphas.push_front(node);
  for (JoinNode *succ : node->successors) {
    succ->alpha_activation(w);
  }
}

// pg 15
// return whether test succeeded or not.
bool const_test_node_activation(ConstTestNode *node, WME *w) {
  std::cerr << __PRETTY_FUNCTION__ << "| node: " << *node << " | wme: " << w
            << "\n";

  // TODO: clean this up, this is a hack.
  // this setting to -1 thing is... terrible.
  if (node->field_to_test != -1) {
    if (w->get_field(node->field_to_test) != node->field_must_equal) {
      return false;
    }
  }

  if (node->output_memory) {
    alpha_memory_activation(node->output_memory, w);
  }
  for (ConstTestNode *c : node->successors) {
    const_test_node_activation(c, w);
  }
  return true;
}

// pg 14
void rete_ctx_add_wme(ReteContext &r, WME *w) {
  r.working_memory.push_back(w);
  const_test_node_activation(r.alpha_top, w);
}

void rete_ctx_remove_wme(ReteContext &r, WME *w) {
  // TODO: actually clear the memory of w.
  r.working_memory.remove(w);
}

// pg 38
void update_new_node_with_matches_from_above(BetaTokensMemory *beta) {
  JoinNode *join = beta->parent;
  std::vector<BetaTokensMemory *> savedListOfSuccessors = join->successors;
  // WTF?
  join->successors = {beta};

  // push alpha memory through join node.
  for (WME *item : join->amem_src->items) {
    join->alpha_activation(item);
  }
  join->successors = savedListOfSuccessors;
}

// pg 34
BetaTokensMemory *build_or_share_beta_memory_node(ReteContext &r,
                                                  JoinNode *parent) {
  // vv TODO: wut? thus looks ridiculous
  for (BetaTokensMemory *succ : parent->successors) {
    return succ;
  }

  BetaTokensMemory *newbeta = new BetaTokensMemory;
  r.betamemories.push_back(newbeta);
  newbeta->parent = parent;
  fprintf(stderr, "%s newBeta: %p | parent: %p\n", __FUNCTION__, newbeta,
          newbeta->parent);
  // newbeta->successors = nullptr;
  // newbeta->items = nullptr;
  parent->successors.push_back(newbeta);
  update_new_node_with_matches_from_above(newbeta);
  return newbeta;
}

// pg 34
JoinNode *build_or_share_join_node(ReteContext &r, BetaTokensMemory *bmem,
                                   AlphaWMEsMemory *amem,
                                   std::vector<TestAtJoinNode> &tests) {
  // bmem can be nullptr in top node case.
  // assert(bmem != nullptr);
  assert(amem != nullptr);

  JoinNode *newjoin = new JoinNode;
  r.joinnodes.push_back(newjoin);
  newjoin->bmem_src = bmem;
  newjoin->tests = tests;
  newjoin->amem_src = amem;
  amem->successors.push_front(newjoin);
  if (bmem) {
    bmem->successors.push_back(newjoin);
  }
  return newjoin;
}

// --- RETE NEGATED CONDITION SUPPORT ---
// --- RETE NEGATED CONDITION SUPPORT ---
// --- RETE NEGATED CONDITION SUPPORT ---
// --- RETE NEGATED CONDITION SUPPORT ---
// --- RETE NEGATED CONDITION SUPPORT ---

// Pg 41.
struct NegativeJoinResult {
  Token *owner; // the token inside whose memory this join result resides.
  WME *wme; // the WME that matches |owner|.

  NegativeJoinResult(Token *owner, WME *wme) : owner(owner), wme(wme) {}
};


// Pg 41
// combination of β memory (hangs on to Token)
// and join node (hangs onto α memory which we need to negate)
struct NegativeNode {
  std::list<Token *> items; // like β
  AlphaWMEsMemory * amem; // like join node.
  std::list<TestAtJoinNode> tests; // tests to run.
  std::list<BetaTokensMemory *> successors; // inferred; is it true that this will always have β node as children?

  // activation wrt WME / left activation / α activation.
  void alpha_activation(Token *t, WME *wme) {
    // vv uh oh, the problem is this is not a beta node.
    // Token *newToken = new Token(this, t, wme); 
    Token *newToken = nullptr;
    this->items.push_front(newToken);

    // compute new join results.
    // TODO: generalize perform_join_tests to work on any type of node.
    if (perform_join_tests(this->tests, newToken, amem->items)) {
      NegativeJoinResult *jr = new NegativeJoinResult(newToken, wme);
      newToken->joinResults.push_front(jr);
      wme->negativeJoinResults.push_front(jr);
    }

    // if join results empty, then inform children.
    if (newToken->joinResults.size() == 0) {
      // what is children?
      for (BetaTokensMemory *succ : successors) {
        // TODO: need a custom WME to indicate negated condition.
        succ->join_activation(newToken, nullptr);
      }
    }
  }
  
};






// --- RETE FRONTEND ---
// --- RETE FRONTEND ---
// --- RETE FRONTEND ---
// --- RETE FRONTEND ---

// inferred from discussion
enum FieldType { Const = 0, Var = 1 };

// inferred from discussion
struct Field {
  FieldType type;
  // TODO: review this dubious code.
  WME::FieldValueT v;

  static Field var(WME::FieldValueT name) {
    Field f;
    f.type = FieldType::Var;
    f.v = name;
    return f;
  }

  static Field constant(WME::FieldValueT name) {
    Field f;
    f.type = FieldType::Const;
    f.v = name;
    return f;
  }
};

// inferred from discussion
struct Condition {
  Field attrs[(int)WME::NFIELDS];
  Condition(Field lhs, Field kind, Field rhs0, Field rhs1) {
    attrs[0] = lhs;
    attrs[1] = kind;
    attrs[2] = rhs0;
    attrs[3] = rhs1;
  }
};

// implicitly defined on pg 35
void lookup_earlier_cond_with_field(const std::vector<Condition> &earlierConds,
                                    WME::FieldValueT v, int *i, int *f2) {
  *i = earlierConds.size() - 1;
  *f2 = -1;

  for (auto it = earlierConds.rbegin(); it != earlierConds.rend(); ++it) {
    for (int j = 0; j < (int)WME::NFIELDS; ++j) {
      if (it->attrs[j].type != FieldType::Var)
        continue;
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
std::vector<TestAtJoinNode>
get_join_tests_from_condition(ReteContext &_, Condition c,
                              const std::vector<Condition> &earlierConds) {
  std::vector<TestAtJoinNode> result;

  for (int f = 0; f < (int)WME::NFIELDS; ++f) {
    if (c.attrs[f].type != FieldType::Var)
      continue;
    // each occurence of variable v
    const WME::FieldValueT v = c.attrs[f].v;
    int i, f2;
    lookup_earlier_cond_with_field(earlierConds, v, &i, &f2);
    // nothing found
    if (i == -1) {
      assert(f2 == -1);
      continue;
    }
    assert(i != -1);
    assert(f2 != -1);
    TestAtJoinNode test;
    test.field_of_arg1 = (WME::FieldKindT)f;
    test.ix_in_token_of_arg2 = i;
    test.field_of_arg2 = (WME::FieldKindT)f2;
    result.push_back(test);
  }
  return result;
};

// page 36
ConstTestNode *build_or_share_constant_test_node(ReteContext &r,
                                                 ConstTestNode *parent,
                                                 WME::FieldKindT f,
                                                 WME::FieldValueT sym) {
  assert(parent != nullptr);
  // look for pre-existing node
  for (ConstTestNode *succ : parent->successors) {
    if (succ->field_to_test == f && succ->field_must_equal == sym) {
      return succ;
    }
  }
  // build a new node
  ConstTestNode *newnode = new ConstTestNode(f, sym, nullptr);
  ;
  r.consttestnodes.push_back(newnode);
  fprintf(stderr, "%s newconsttestnode: %p\n", __FUNCTION__, newnode);
  parent->successors.push_back(newnode);
  // newnode->field_to_test = f; newnode->field_must_equal = sym;
  // newnode->output_memory = nullptr;
  // newnode->successors = nullptr;
  return newnode;
}

// implied in page 35: build_or_share_alpha_memory.
bool wme_passes_constant_tests(WME *w, Condition c) {
  for (int f = 0; f < (int)WME::NFIELDS; ++f) {
    if (c.attrs[f].type != FieldType::Const)
      continue;
    if (c.attrs[f].v != w->fields[f])
      return false;
  }
  return true;
}

// pg 35: dataflow version
AlphaWMEsMemory *build_or_share_alpha_memory_dataflow(ReteContext &r,
                                                      Condition c) {
  ConstTestNode *currentNode = r.alpha_top;
  for (int f = 0; f < (int)WME::NFIELDS; ++f) {
    if (c.attrs[f].type != FieldType::Const)
      continue;
    const WME::FieldValueT sym = c.attrs[f].v;
    currentNode = build_or_share_constant_test_node(r, currentNode,
                                                    (WME::FieldKindT)f, sym);
  }

  if (currentNode->output_memory != nullptr) {
    return currentNode->output_memory;
  } else {
    assert(currentNode->output_memory == nullptr);
    currentNode->output_memory = new AlphaWMEsMemory;
    r.alphamemories.push_back(currentNode->output_memory);
    // initialize AM with any current WMEs
    for (WME *w : r.working_memory) {
      // check if wme passes all constant tests
      if (wme_passes_constant_tests(w, c)) {
        alpha_memory_activation(currentNode->output_memory, w);
      }
    }
    return currentNode->output_memory;
  }
};

// page 36: hash version
AlphaWMEsMemory *build_or_share_alpha_memory_hashed(ReteContext &r,
                                                    Condition c) {
  assert(false && "unimplemented");
};

// pg 37
// - inferred type of production node:
ProductionNode *rete_ctx_add_production(ReteContext &r,
                                        const std::vector<Condition> &lhs,
                                        ProductionNode::CallbackT callback,
                                        std::string rhs) {
  // pseudocode: pg 33
  // M[1] <- dummy-top-node
  // build/share J[1] (a succ of M[1]), the join node for c[1]
  // for i = 2 to k do
  //     build/share M[i] (a succ of J[i-1]), a beta memory node
  //     build/share J[i] (a succ of M[i]), the join node for ci
  // make P (a succ of J[k]), the production node
  assert(lhs.size() > 0);
  std::vector<Condition> earlierConds;

  std::vector<TestAtJoinNode> tests =
      get_join_tests_from_condition(r, lhs[0], earlierConds);
  AlphaWMEsMemory *am = build_or_share_alpha_memory_dataflow(r, lhs[0]);

  BetaTokensMemory *currentBeta = nullptr;
  JoinNode *currentJoin = build_or_share_join_node(r, currentBeta, am, tests);
  earlierConds.push_back(lhs[0]);

  // TODO: why not start with 0?
  for (int i = 1; i < (int)lhs.size(); ++i) {
    // get the current beat memory node M[i]
    currentBeta = build_or_share_beta_memory_node(r, currentJoin);
    // get the join node J[i] for condition c[u[
    tests = get_join_tests_from_condition(r, lhs[i], earlierConds);
    am = build_or_share_alpha_memory_dataflow(r, lhs[i]);
    currentJoin = build_or_share_join_node(r, currentBeta, am, tests);
    earlierConds.push_back(lhs[i]);
  }

  // build a new production node, make it a succ of current node
  ProductionNode *prod = new ProductionNode;
  r.productions.push_back(prod);
  prod->parent = currentJoin; // currentJoin is guaranteed to be valid
  fprintf(stderr, "%s prod: %p | parent: %p\n", __FUNCTION__, prod,
          prod->parent);
  prod->callback = callback;
  prod->rhs = rhs;
  currentJoin->successors.push_back(prod);
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

const int ADD_OP_KIND = '+';
const int INT_OP_KIND = 'i';

ReteContext *toRete(mlir::FuncOp f, mlir::IRRewriter rewriter) {
  ReteContext *ctx = new ReteContext();
  assert(f.getBlocks().size() == 1 && "currently do not handle branching");

  {
    // add conditions
    const int sym_add_lhs = 0;
    const int sym_add_rhs1 = 1;
    const int sym_add_rhs2 = 2;
    const int sym_rhs1_val = 3;
    const int sym_rhs2_val = 4;
    std::vector<Condition> addConditions;
    addConditions.push_back(Condition(
        Field::var(sym_add_lhs), Field::constant(ADD_OP_KIND),
        Field::var(sym_add_rhs1), Field::var(sym_add_rhs2)));
    addConditions.push_back(Condition(
        Field::var(sym_add_rhs1), Field::constant(INT_OP_KIND),
        Field::var(sym_rhs1_val), Field::constant(0)));
    addConditions.push_back(Condition(
        Field::var(sym_add_rhs2), Field::constant(INT_OP_KIND),
        Field::var(sym_rhs2_val), Field::constant(0)));

    rete_ctx_add_production(
        *ctx, addConditions,
        [&](Token *t, WME *w) {
          mlir::SmallVector<WME *> args;
          // mlir::SmallVector<mlir::Value> guids;
          // mlir::SmallVector<mlir::Value> insts;
          llvm::errs() << "*** token->ix: " << t->token_chain_ix << "| ***\n";
          for (int i = 0; i <= t->token_chain_ix; ++i) {
            WME *arg = t->index(i);
            args.push_back(arg);
            llvm::errs() << "*** found constant folding opportunity [" << i
                         << " kind[" << (char)(arg->fields[1]) << "]"
                         << "***\n";
          }

          WME *wme = new WME;
          wme->fields[0] = args[0]->fields[0]; // we are replacing the add op's result.
          wme->fields[1] = INT_OP_KIND;
          wme->fields[2] = args[1]->fields[2] + args[2]->fields[2]; // our value is the sum of the LHS value and the RHS value.
          wme->fields[3] = 0;
          // probably incorrect to remove this? 
          rete_ctx_remove_wme(*ctx, args[0]);
          rete_ctx_add_wme(*ctx, wme);
        },
        "add_const_fold");
  }


  std::map<mlir::Operation *, int64_t> val2guid;
  for (mlir::Operation &op : f.getBlocks().front()) {
    if (AsmAddOp add = mlir::dyn_cast<AsmAddOp>(op)) {
      WME *wme = new WME;
      const int guid = ctx->gensym();
      val2guid[add] = guid; 
      wme->fields[0] = guid;
      wme->fields[1] = ADD_OP_KIND;
      mlir::Operation *lhs = add.lhs().getDefiningOp();
      mlir::Operation *rhs = add.rhs().getDefiningOp();
      assert(val2guid.count(lhs));
      assert(val2guid.count(rhs));
      wme->fields[2] = val2guid[lhs];
      wme->fields[3] = val2guid[rhs];
      rete_ctx_add_wme(*ctx, wme);
      continue;
    }
    if (AsmIntOp i = mlir::dyn_cast<AsmIntOp>(op)) {
      WME *wme = new WME;
      const int guid = ctx->gensym();
      val2guid[i] = guid; 
      wme->fields[0] = guid;
      wme->fields[1] = INT_OP_KIND;
      wme->fields[2] = i.getValue();
      wme->fields[3] = 0;
      rete_ctx_add_wme(*ctx, wme);
      continue;
    }
    if (mlir::ReturnOp ret = mlir::dyn_cast<mlir::ReturnOp>(op)) {
      // do nothing
      continue;
    }
    llvm::errs() << op << "\n";
    assert(false && "unknown operation to RETE");
  }
  return ctx;
};

mlir::FuncOp fromRete(mlir::MLIRContext *mlir_ctx, mlir::ModuleOp m,
                      ReteContext *rete_ctx, mlir::IRRewriter &rewriter) {
  mlir::FunctionType fnty = rewriter.getFunctionType({}, {});
  rewriter.setInsertionPointToStart(m.getBody());
  mlir::FuncOp fn = rewriter.create<mlir::FuncOp>(rewriter.getUnknownLoc(),
                                                  "rewritten_fn", fnty);
  fn.setPrivate();
  {
    mlir::Block *entry = fn.addEntryBlock();
    rewriter.setInsertionPoint(entry, entry->begin());
  }

  std::map<int64_t, mlir::Value> guid2val;

  // TODO, HACK: reverse the working memory so we get the newest replacement of each instruction..
  // rete_ctx->working_memory.reverse();

  bool done = false;
  while (!done) {
    done = true;
    // loop over instructions
    for (WME *wme : rete_ctx->working_memory) {
      // materialize instructions
      if (guid2val.count(wme->fields[0])) {
        // have already mateialized.
        continue;
      }

      const int kind = (int)reinterpret_cast<int64_t>(wme->fields[1]);
      if (kind == ADD_OP_KIND) {
        if (!guid2val.count(wme->fields[3])) {
          continue;
        }
        if (!guid2val.count(wme->fields[2])) {
          continue;
        }
        done = false;

        mlir::Value lhs = guid2val[wme->fields[2]];
        mlir::Value rhs = guid2val[wme->fields[3]];
        AsmAddOp add =
            rewriter.create<AsmAddOp>(rewriter.getUnknownLoc(), lhs, rhs);
        guid2val[wme->fields[0]] = add.getResult();
        rewriter.setInsertionPointAfter(add);
        continue;
      }

      if (kind == INT_OP_KIND) {
        done = false;
        const int value = int(wme->fields[2]);
        AsmIntOp i = rewriter.create<AsmIntOp>(rewriter.getUnknownLoc(), value);
        rewriter.setInsertionPointAfter(i);
        guid2val[wme->fields[0]] = i.getResult();
        continue;
      }

      assert(false && "unreachable");

    } // end wme loop.
  }   // end done loop.
  rewriter.setInsertionPointToEnd(&*fn.getBlocks().begin());
  rewriter.create<mlir::ReturnOp>(rewriter.getUnknownLoc());
  return fn;
};

struct ReteOptimizationPass : public mlir::Pass {
  ReteOptimizationPass()
      : mlir::Pass(mlir::TypeID::get<ReteOptimizationPass>()){};
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

    mod.walk([&](mlir::FuncOp fn) {
      // TODO: to_rete should not need rewriter!
      ReteContext *rete_ctx = toRete(fn, rewriter);
      // fn.erase();
      mlir::FuncOp newFn = fromRete(fn.getContext(), mod, rete_ctx, rewriter);
      // (void)newFn;
    });
  }
};

// === GREEDY PATTERN DRIVER PASS ===
// === GREEDY PATTERN DRIVER PASS ===
// === GREEDY PATTERN DRIVER PASS ===

class FoldAddPattern : public mlir::OpRewritePattern<AsmAddOp> {

public:
  FoldAddPattern(mlir::MLIRContext *context)
      : OpRewritePattern<AsmAddOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult
  matchAndRewrite(AsmAddOp add,
                  mlir::PatternRewriter &rewriter) const override {
    AsmIntOp lhs = add.lhs().getDefiningOp<AsmIntOp>();
    AsmIntOp rhs = add.rhs().getDefiningOp<AsmIntOp>();
    if (!lhs || !rhs) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<AsmIntOp>(add, lhs.getValue() + rhs.getValue());
    return mlir::success();
  }
};

struct GreedyOptimizationPass : public mlir::Pass {
  GreedyOptimizationPass()
      : mlir::Pass(mlir::TypeID::get<GreedyOptimizationPass>()){};
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
  mlir::registerPass("bench-greedy",
                     "Rewrite using greedy pattern rewrite driver",
                     []() -> std::unique_ptr<::mlir::Pass> {
                       return std::make_unique<GreedyOptimizationPass>();
                     });

  mlir::registerPass("bench-rete", "Rewrite using RETE algorithm",
                     []() -> std::unique_ptr<::mlir::Pass> {
                       return std::make_unique<ReteOptimizationPass>();
                     });

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<AsmDialect>();
  return failed(mlir::MlirOptMain(argc, argv, "Hoopl optimization drver",
                                  registry, true));
}
