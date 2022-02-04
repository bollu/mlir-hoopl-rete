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

# Benchmarks

## `test/rand-program-seed-0.mlir`

##### `GreedyPatternRewriter`


```
bollu@scheme ~/w/1/test (master)> ./generate-rand-program.py && 
  make -C ~/work/1-hoopl/build/release/ && 
  perf stat  ~/work/1-hoopl/build/release/bin/hoopl --bench-greedy 
  ~/work/1-hoopl/test/rand-program-seed-0.mlir > /dev/null

            504.71 msec task-clock                #    0.999 CPUs utilized          
                 0      context-switches          #    0.000 /sec                   
                 0      cpu-migrations            #    0.000 /sec                   
            20,840      page-faults               #   41.291 K/sec                  
     1,655,810,133      cycles                    #    3.281 GHz                    
     2,515,295,053      instructions              #    1.52  insn per cycle         
       536,519,475      branches                  #    1.063 G/sec                  
         4,923,680      branch-misses             #    0.92% of all branches        

       0.505344752 seconds time elapsed

       0.464690000 seconds user
       0.039756000 seconds sys
```


```
  10.52%  hoopl    [kernel.vmlinux]                      [k] syscall_exit_to_user_mode
   4.72%  hoopl    [kernel.vmlinux]                      [k] entry_SYSCALL_64
   4.05%  hoopl    [kernel.vmlinux]                      [k] syscall_return_via_sysret
   2.53%  hoopl    [kernel.vmlinux]                      [k] n_tty_write
   2.16%  hoopl    [kernel.vmlinux]                      [k] _raw_spin_lock_irqsave
   1.71%  hoopl    [kernel.vmlinux]                      [k] preempt_count_add
   1.46%  hoopl    [kernel.vmlinux]                      [k] tty_write
   1.30%  hoopl    [kernel.vmlinux]                      [k] _raw_spin_unlock_irqrestore
   1.28%  hoopl    [kernel.vmlinux]                      [k] preempt_count_sub
   1.26%  hoopl    ld-2.33.so                            [.] do_lookup_x
   1.25%  hoopl    [kernel.vmlinux]                      [k] try_to_wake_up
   1.17%  hoopl    libLLVMSupport.so.13git               [.] llvm::StringMapImpl::LookupBucketFor
   1.05%  hoopl    [kernel.vmlinux]                      [k] queue_work_on
   1.05%  hoopl    [kernel.vmlinux]                      [k] vfs_write
   1.02%  hoopl    [kernel.vmlinux]                      [k] _raw_spin_lock
   1.00%  hoopl    [kernel.vmlinux]                      [k] native_queued_spin_lock_slowpath
   0.90%  hoopl    libMLIRTransformUtils.so.13git        [.] mlir::applyPatternsAndFoldGreedily
   0.87%  hoopl    libMLIRIR.so.13git                    [.] llvm::DenseMapBase<llvm::DenseMap<mlir::Value, unsigned int, llvm::DenseMapInfo<mlir::Value>, llvm::detail::DenseMapPair<mlir::Value, unsigned int> >, mlir::Value, unsigned int, llvm::DenseMapInfo<mlir::Value>, llvm::detail::DenseMapPair<mlir::Value, unsigned int> >::LookupBucketFor<mlir::Value>
   0.84%  hoopl    [kernel.vmlinux]                      [k] insert_work
   0.84%  hoopl    libMLIRTransformUtils.so.13git        [.] propagateLiveness
   0.82%  hoopl    [kernel.vmlinux]                      [k] update_rq_clock
   0.75%  hoopl    [kernel.vmlinux]                      [k] select_task_rq_fair
   0.75%  hoopl    [kernel.vmlinux]                      [k] __check_object_size
   0.74%  hoopl    [kernel.vmlinux]                      [k] __fget_light
   0.73%  hoopl    libMLIRSupport.so.13git               [.] (anonymous namespace)::ParametricStorageUniquer::getOrCreate
   0.73%  hoopl    [kernel.vmlinux]                      [k] tty_insert_flip_string_fixed_flag
   0.71%  hoopl    [kernel.vmlinux]                      [k] enqueue_entity
   0.70%  hoopl    [kernel.vmlinux]                      [k] __queue_work
   0.70%  hoopl    [kernel.vmlinux]                      [k] enqueue_task_fair
   0.69%  hoopl    [kernel.vmlinux]                      [k] native_sched_clock
   0.67%  hoopl    [kernel.vmlinux]                      [k] __wake_up_common_lock
   0.66%  hoopl    [kernel.vmlinux]                      [k] psi_group_change
   0.66%  hoopl    [kernel.vmlinux]                      [k] apparmor_file_permission
   0.65%  hoopl    [kernel.vmlinux]                      [k] in_lock_functions
   0.64%  hoopl    [kernel.vmlinux]                      [k] __tty_buffer_request_room
```

##### `rete`

```
bollu@scheme ~/w/1/test (master)> ./generate-rand-program.py && 
  make -C ~/work/1-hoopl/build/release/ &&
  perf stat  ~/work/1-hoopl/build/release/bin/hoopl --bench-rete 
  ~/work/1-hoopl/test/rand-program-seed-0.mlir > /dev/null

            841.49 msec task-clock                #    0.999 CPUs utilized          
                 1      context-switches          #    1.188 /sec                   
                 0      cpu-migrations            #    0.000 /sec                   
            49,471      page-faults               #   58.790 K/sec                  
     2,828,704,517      cycles                    #    3.362 GHz                    
     3,741,444,223      instructions              #    1.32  insn per cycle         
       804,469,224      branches                  #  956.007 M/sec                  
         6,608,962      branch-misses             #    0.82% of all branches        

       0.842037753 seconds time elapsed

       0.751051000 seconds user
```

```
   6.96%  hoopl    libc-2.33.so                       [.] _int_malloc
   5.65%  hoopl    hoopl                              [.] AlphaWMEsMemory::alpha_activation
   4.76%  hoopl    hoopl                              [.] fromRete
   3.87%  hoopl    hoopl                              [.] JoinNode::join_activation
   3.11%  hoopl    hoopl                              [.] std::_Hashtable<WME*, WME*, std::allocator<WME*>, std::__detail::_Identity, std::equal_to<WME*>, std::hash<WME*>,
   3.11%  hoopl    hoopl                              [.] toRete
   2.74%  hoopl    ld-2.33.so                         [.] do_lookup_x
   2.26%  hoopl    libMLIRIR.so.13git                 [.] llvm::DenseMapBase<llvm::DenseMap<mlir::Value, unsigned int, llvm::DenseMapInfo<mlir::Value>, llvm::detail::DenseM
   2.23%  hoopl    hoopl                              [.] rete_ctx_add_wme
   2.23%  hoopl    libc-2.33.so                       [.] malloc
   2.20%  hoopl    libstdc++.so.6.0.29                [.] std::_Rb_tree_increment
   2.11%  hoopl    hoopl                              [.] std::_Hashtable<WME*, WME*, std::allocator<WME*>, std::__detail::_Identity, std::equal_to<WME*>, std::hash<WME*>,
   2.07%  hoopl    libc-2.33.so                       [.] malloc_consolidate
   1.99%  hoopl    libLLVMSupport.so.13git            [.] llvm::StringMapImpl::LookupBucketFor
   1.85%  hoopl    libMLIRSupport.so.13git            [.] (anonymous namespace)::ParametricStorageUniquer::getOrCreate
   1.39%  hoopl    ld-2.33.so                         [.] strcmp
   1.32%  hoopl    libstdc++.so.6.0.29                [.] std::_Rb_tree_insert_and_rebalance
   1.22%  hoopl    libc-2.33.so                       [.] _int_free
   1.21%  hoopl    [kernel.vmlinux]                   [k] irqentry_exit_to_user_mode
   0.99%  hoopl    libMLIRIR.so.13git                 [.] llvm::function_ref<bool (mlir::StorageUniquer::BaseStorage const*)>::callback_fn<mlir::StorageUniquer::get<mlir::d
   0.82%  hoopl    libMLIRIR.so.13git                 [.] (anonymous namespace)::OperationVerifier::verifyDominance
   0.79%  hoopl    libLLVMSupport.so.13git            [.] llvm::StringMapImpl::FindKey
   0.79%  hoopl    libMLIRIR.so.13git                 [.] llvm::DenseMap<mlir::Value, unsigned int, llvm::DenseMapInfo<mlir::Value>, llvm::detail::DenseMapPair<mlir::Value,
   0.73%  hoopl    hoopl                              [.] rete_ctx_remove_wme
   0.72%  hoopl    hoopl                              [.] std::_Rb_tree<mlir::Operation*, std::pair<mlir::Operation* const, long>, std::_Select1st<std::pair<mlir::Operation
   0.70%  hoopl    libLLVMSupport.so.13git            [.] llvm::SourceMgr::SrcBuffer::getLineNumberSpecialized<unsigned int>
   0.68%  hoopl    libMLIRIR.so.13git                 [.] mlir::Operation::dropAllReferences
```


# Build instructions

```bash
$ 
mkdir -p llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm/ \
 -DLLVM_ENABLE_PROJECTS=mlir \
 -DLLVM_TARGETS_TO_BUILD="X86" \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_LLD=ON
ninja
$
mkdir -p build 
cd build
CMAKE_PREFIX_PATH=`pwd`/../llvm-project/build cmake -GNinja ../   -DBUILD_SHARED_LIBS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja
```


# Emacs nonsense

- use `flymke-proc-compile` to setup `ninja` launch
