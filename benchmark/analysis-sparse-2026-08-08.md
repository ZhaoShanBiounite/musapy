# Benchmark 分析（2026-08-08）：sparse 套件（v0.3 Phase 6）

分支 `feat/v0.3-musax-ffi`，**MTT S4000 真机**（mp_22, 56 CUs, 47.9 GB VRAM），
release 构建。数据：`benchmark/bench_sparse.py --iters 15 --n 2000`。

## 1. 真机性能数据（2000×2000 稀疏矩阵）

| density | nnz | spmv(ms) | spmm k=4(ms) | spmv 有效带宽(GB/s) |
|---|---|---|---|---|
| 0.01 | 40,000 | 0.72 | 0.06 | 0.4-0.7 |
| 0.10 | 400,000 | 0.81-0.89 | 0.10-0.16 | 4.0-5.4 |
| 0.50 | 2,000,000 | 0.96-1.08 | 0.33-0.61 | 16.7-22.3 |

## 2. 归因

1. **spmv 低密度下延迟 ~0.72ms 不随 nnz 明显变化**：小矩阵（2000²）下
   两段式（查询→workspace 分配→计算）+ handle/描述符创建开销主导，
   与 nnz 关系弱。带宽随 density 上升（大 nnz 摊薄固定开销）。
2. **spmm 明显快于 spmv**（同 nnz 下 10×）：musparse SpMM kernel 对 k=4
   列稠密矩阵批量处理高效；spmv 每次只有 1 列向量，访存效率低。
3. **有效带宽上限 ~22 GB/s**（density=0.5）：CSR 非合并访问（indices 随机）
   特性，与 gather 的随机访问带宽特征一致（64M 下 ~64 GB/s，见 repo.md）。

## 3. 已知限制与后续

- `toarray` 走 D2H→host 构建→H2D，未纳入吞吐基准（正确性优先）。
- spmv 单次调用含描述符创建/销毁（每调用 create/destroy），大矩阵高频调用
  可考虑描述符缓存（仿 math_handle plan 池）——后续可选优化。
- 非方阵、空矩阵、f32/f64 数值均已对照 NumPy 通过（test_sparse.py 19 用例）。

## 4. 门禁

- pytest 536 passed（+19 sparse）· cargo test 301 passed · mock 模式 sparse 19 passed

## 5. P-A3 优化（2026-08-08，追加）

**描述符缓存**：spmv/spmm 的 `musparseSpMatDescr_t` 改由 `math_handle` 池化缓存
（`MusparseSpMatSpec` 键 = 三 buffer 指针 + shape + nnz + dtype，仿 mufft_plans），
消除每调用 create/destroy 固定开销。

| 场景 | 优化前 | 优化后 | 收益 |
|---|---|---|---|
| spmv 2000² d=0.01 | 0.67 ms | **0.061 ms** | ✅ **11×** |
| spmm 2000² d=0.01 k=4 | 0.055 ms | 0.063 ms | ✅ 已近 launch 地板 |

- `with_musparse_csr`（math_handle.rs）：句柄 + 描述符懒创建/缓存 + SetStream + 闭包
- `DeferredDestroy` 加 `MusparseSpMat` 变体，`evict_device` 统一入延迟销毁队列
- DnVec/DnMat 仍每调用创建（轻量描述符，缓存收益低）
- 修复 test_handle_cycle_mem_flat 顺序耦合：workspace 桶缓存被前置测试留下、
  evict 回收导致 after<before，断言改为「不增长」（泄漏仍可抓）
