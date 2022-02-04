// RUN: mlir-opt %s  -allow-unregistered-dialect -test-pdl-bytecode-pass -split-input-file | FileCheck %s

// -----

//===----------------------------------------------------------------------===//
// Asm add rewrite to write add (int 0) a -> a
//===----------------------------------------------------------------------===//

module @patterns {
  // fc_fwd
  pdl.pattern : benefit(1) {
    %c0_type = pdl.type
    %cr_type = pdl.type
    %cr = pdl.operand : %cr_type

    %c0_attr = pdl.attribute 0 : i32
    // TODO: is pdl.operation  allowed to have empty arg list?
    // %op0 = pdl.operation "asm.int" () {"value" = %c0_attr} -> (%c0_type : !pdl.type)
    %op0 = pdl.operation "asm.int" {"value" = %c0_attr} -> (%c0_type : !pdl.type)
    %val0 = pdl.result 0 of %op0
    %opadd = pdl.operation "asm.add" (%val0, %cr : !pdl.value, !pdl.value) -> (%c0_type : !pdl.type)

    pdl.rewrite %opadd {
      // %op1 = pdl.operation "kernel.FcFwd" (%rxact, %weight : !pdl.value, !pdl.value) -> (%out_type : !pdl.type)
      %val1 = pdl.result 1 of %opadd
      pdl.replace %op0 with (%val1 : !pdl.value)  
    }
  }
}

module @ir attributes { test.mlp_split } {
  func @main(%r: i32) -> (i32) {
    %c0 = "asm.int"() { value = 0 : i32} : () -> (i32)
    %add = "asm.add"(%c0, %r) : (i32, i32) -> (i32)
    return %add : i32
  }
}

