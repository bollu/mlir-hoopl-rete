func @main() {
  %x = constant 1 : i64
  %y = constant 2 : i64
  %z = addi %x, %y : i64
  return
}

