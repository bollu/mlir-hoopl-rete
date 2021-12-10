func @main() {
  %x = dead
  %y = dead
  %z = asm.add %x, %y
  %w = dead // should be removed by DCE
  return
}
