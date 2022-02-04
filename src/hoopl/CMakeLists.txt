# https://cmake.org/cmake/help/latest/prop_tgt/CXX_STANDARD.html
set(CMAKE_CXX_STANDARD 17)

# LLVM components
# https://github.com/llvm/llvm-project/blob/a93514abf2d843339878d0607e2f5fd2a787bf72/mlir/examples/toy/Ch7/CMakeLists.txt
set(LLVM_LINK_COMPONENTS
  Core
  MCParser  # found from llvm-exegesis.
  Support
  native # fond from llvm-exegesis
  nativecodegen
  OrcJIT
  ExecutionEngine
  Interpreter
  X86 #HACK?
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
add_llvm_executable(hoopl main.cpp)

llvm_update_compile_flags(hoopl)
target_include_directories(hoopl PRIVATE ../immer/)
target_link_libraries(hoopl PRIVATE
  ${dialect_libs}
  #       MLIRGRIN
  #       MLIRPointer
        ${conversion_libs}
        MLIRTranslation
        MLIROptLib
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        # MLIRTargetLLVMIR
        MLIRTransforms
        MLIRSupport)
        # cgraph gvc)



mlir_check_link_libraries(hoopl)

