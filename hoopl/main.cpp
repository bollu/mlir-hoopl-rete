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

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

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

class PtrDialect : public mlir::Dialect {
public:
  explicit PtrDialect(mlir::MLIRContext *ctx);
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
  // mlir::Attribute parseAttribute(mlir::DialectAsmParser &parser,
  //                               Type type) const override;
  // void printAttribute(Attribute attr,
  //                     DialectAsmPrinter &printer) const override;
  static llvm::StringRef getDialectNamespace() { return "ptr"; }
};

class PtrType : public mlir::Type {
public:
  /// Inherit base constructors.
  using mlir::Type::Type;

  /// Support for PointerLikeTypeTraits.
  using mlir::Type::getAsOpaquePointer;
  static PtrType getFromOpaquePointer(const void *ptr) {
    return PtrType(static_cast<ImplType *>(const_cast<void *>(ptr)));
  }
  /// Support for isa/cast.
  static bool classof(Type type);
  PtrDialect &getDialect();
};


using namespace llvm;
using namespace llvm::orc;

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  // mlir::registerInlinerPass();
  // mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
//   registerRgnCSEPass();
  // mlir::registerAffinePasses();
  // mlir::registerAffineLoopFusionPass();
  // mlir::registerConvertStandardToLLVMPass();
  // mlir::registerConvertAffineToStandardPass();
  // mlir::registerSCFToStandardPass();

  // mlir::standalone::registerWorkerWrapperPass();
  // mlir::unif::registerUnifierPass();
  // mlir::standalone::registerLowerHaskPass();
  // mlir::standalone::registerLowerLeanPass();
  // mlir::standalone::registerLowerLeanRgnPass();
  // mlir::standalone::registerFuseRefcountPass();
  // mlir::ptr::registerLowerPointerPass();
  // registerLZJITPass();
  // registerLZDumpLLVMPass();
  // registerLzInterpretPass();
  // registerLzLazifyPass();
  // registerLowerRgnPass();
  // registerOptimizeRgnPass();
  // mlir::standalone::registerHaskCanonicalizePass();
  // mlir::standalone::registerWrapperWorkerPass();

  // mlir::lambdapure::registerLambdapureToLeanLowering();
  // mlir::lambdapure::registerReferenceRewriterPattern();
  // mlir::lambdapure::registerDestructiveUpdatePattern();

  // mlir::registerScfToRgnPass();  

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::LLVM::LLVMDialect>();
  // registry.insert<mlir::standalone::HaskDialect>();
  // registry.insert<mlir::grin::GRINDialect>();
  // registry.insert<mlir::lambdapure::LambdapureDialect>();
  // registry.insert<mlir::ptr::PtrDialect>();
  // registry.insert<mlir::unif::UnificationDialect>();
  // registry.insert<RgnDialect>();

  // registry.insert<mlir::StandardOpsDialect>();
  // registry.insert<mlir::AffineDialect>();
  // registry.insert<mlir::scf::SCFDialect>();

  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);

  return failed(
      mlir::MlirOptMain(argc, argv, "Hoopl optimization drver", registry, true));
}

