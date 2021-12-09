// func @main() -> i64 {
//   %x = constant 1 : i64
//   %y = constant 2 : i64
//   %z = addi %x, %y : i64
//   return %z : i64
// }


func @main()  {
  %x = asm.int 1 
  %y = asm.int 2
  %z = asm.add %x, %y
  %w = asm.add %z, %z
  return 
}

