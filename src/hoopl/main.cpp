#include<stdio.h>
#include<graphviz/cgraph.h>
#include<graphviz/gvc.h>
#include<sstream>
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
#include <unordered_set>
#include <optional>

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

// Immer
#include "immer/vector.hpp"
#include "immer/flex_vector.hpp"
#include "immer/map.hpp"
#include "immer/box.hpp"

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



// === HOOPL ===
// === HOOPL ===
// === HOOPL ===
// === HOOPL ===
// === HOOPL ===
// === HOOPL ===
// === HOOPL ===
// === HOOPL ===

// some kind of unique label ID thing.
struct BBLabel {
  int guid;
  BBLabel(int guid) : guid(guid) {}

  bool operator == (const BBLabel &other) const {
    return this->guid == other.guid;
  }

  bool operator  < (const BBLabel &other) const {
    return this->guid < other.guid;
  }
};

namespace std { 
template<>
struct hash<BBLabel> {
    std::size_t operator()(const BBLabel &x) const {
        return std::hash<int>{}(x.guid);
    }
};

} // namespace std


enum class OpType {
  ADD,
  CONST,
};

struct HooplNode {
  bool terminator;
  static const int MAX_SUCCESSORS = 2;
  int nextscount;
  BBLabel nexts[MAX_SUCCESSORS];
  static const int MAX_ARGUMENTS = 3;
  immer::box<HooplNode> arguments [MAX_ARGUMENTS];
  OpType type;
};


// virtual interface for fact lattices.
struct Fact {
  virtual Fact *bottom() = 0;
  virtual Fact *unite(Fact *a, Fact *b) = 0;
};

template <typename F, typename  T>
struct ForwardPass {
    virtual F bottom() = 0;
    virtual F unite(F oldfact, F newfact, bool *didChange = nullptr) = 0; // returns if oldfact \/ newfact == oldfact.
    virtual F forwardTransferInst(F fact, immer::box<HooplNode> n) = 0;
    virtual immer::map<BBLabel, F> forwardTransferTerminator(F fact, immer::box<HooplNode> n) = 0;
    virtual std::optional<immer::box<HooplNode>> rewrite(F f, immer::box<HooplNode> n) = 0; 
};

struct HooplBlock {
  immer::flex_vector<immer::box<HooplNode> > nodes;
};


// control flow graph. 
// (region?)
struct Graph {
  immer::box<HooplBlock> entry;
  immer::map<BBLabel, immer::box<HooplBlock>> label2block; 
  // immer::box<HooplBlock> exit; // why should it be SESE?
};

// we only do this per function
// we shall eventually handle nested CFGs
Graph functionToHoopl(mlir::FuncOp func) {
  Graph CFG;
  std::map<mlir::Block *, immer::box<HooplBlock>> mlir2hooplbb;
  int bbgensym = 0;

  for(mlir::Block &bb : func.getRegion().getBlocks()) {
    immer::box<HooplBlock> hbb;    
    mlir2hooplbb[&bb] = hbb;
    BBLabel label(++bbgensym);
    CFG.label2block.insert({label, hbb});
  }
  auto entry = mlir2hooplbb.find(&*func.getBlocks().begin());
  assert(entry != mlir2hooplbb.end());
  CFG.entry = entry->second;
  return CFG;
}

struct HooplOptimizationPass : public mlir::Pass {
  HooplOptimizationPass()
      : mlir::Pass(mlir::TypeID::get<HooplOptimizationPass>()){};
  mlir::StringRef getName() const override { return "ReteOptimization"; }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    auto newInst = std::make_unique<HooplOptimizationPass>(
        *static_cast<const HooplOptimizationPass *>(this));
    newInst->copyOptionValuesFrom(this);
    return newInst;
  }

  void runOnOperation() override {
    mlir::ModuleOp mod = mlir::cast<mlir::ModuleOp>(this->getOperation());
    mlir::IRRewriter rewriter(mod.getContext());

    mod.walk([&](mlir::FuncOp fn) {
      // TODO: to_rete should not need rewriter!
      // ReteContext *rete_ctx = toRete(fn, rewriter);
      // mlir::FuncOp newFn = fromRete(fn.getContext(), mod, rete_ctx, rewriter);
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

  mlir::registerPass("bench-hoopl", "Rewrite using RETE algorithm",
                     []() -> std::unique_ptr<::mlir::Pass> {
                       return std::make_unique<HooplOptimizationPass>();
                     });

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::LLVM::LLVMDialect>();
  registry.insert<AsmDialect>();
  return failed(mlir::MlirOptMain(argc, argv, "Hoopl optimization drver",
                                  registry, true));
}
