# MLIR, Hoopl, Rete

That's a very confused title. What's this repo about?
- MLIR needs the ability to perform fast pattern matching and rewrites on the IR.
- [rete](https://en.wikipedia.org/wiki/Rete_algorithm) is an ancient algorithm from the time of Old AI to pattern match
  and rewrite databases of 'facts'. 
- [Hoopl](https://github.com/haskell/hoopl) is a technique (originally developed for the Haskell compiler, GHC) to interleave
  dataflow analysis and rewrites on a nested CFG.
- It appears that there's an opportunity here. Shove MLIR (which is a CFG, nested by regions) into a Rete network, and add
  pattern matchers and rewrites to "mine facts" from the IR. That this is safe to do is proven in the [Hoopl paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/hoopl-haskell10.pdf).
- If this works, we get a clean, decades old algorithm to implement MLIR *analyses and rewrites* .
- Only problem: I implemented the naive `rete` algorithm. [the benchmarks aren't so hot](https://github.com/bollu/mlir-hoopl-rete/pull/2).
- I have ideas on how to improve the performance (use hashing within the rete `join` nodes, don't pay `O(n^2)` cost to match uses with defs).
  But this needs more time. I'll probably grab some time during a retreat `:)`.

