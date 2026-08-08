# Changelog

本仓库自 v0.2.0-alpha 起的变更记录（按类型分组，时间序）。

## [v0.3.0-alpha] — 2026-08-08

### Features

- **v0.3 Phase 1** — MUSA-X FFI 基础设施（P1.1–P1.7，ADR-003 003-D1/D2）
- **v0.3 Phase 2** — linalg A：matmul/dot/solve（GPU-only）+ benchmark 扩展
- **v0.3 Phase 3** — linalg B 分解类算子：lu/qr/svd（P3.1–P3.6）
- **v0.3 Phase 4** — random 套件：rand/randn/uniform/normal/bernoulli（P4.1–P4.7）
- **v0.3 Phase 5** — fft 套件 + complex 落地（P5.1–P5.4/P5.7，ADR-003 003-D5/D7）
- **v0.3 Phase 6** — sparse 套件：csr_matrix + spmv/spmm/toarray（P6.1/P6.3/P6.4/P6.6）
- **v0.3 Phase 7** — reduction 补全：axis=tuple + 复数 sum/mean/prod（P7.1–P7.3）
- **v0.3 Phase 8** — 高级索引：boolean mask + fancy indexing（P8.1–P8.4）
- **dtype 字符串语法** — 全部 dtype 参数接受字符串短别名（`'f32'`/`'i64'`/`'c64'`/
  `'b1'` 等）或全名（`'float32'`），`a.dtype == 'f32'` 可互比；兼容 `ms.float32`
  常量（`63e621c`）

### Performance

- **P0** — solve 奇异检测设备端化（extract_diag kernel，solve(1024) -55%）
- **P1** — elementwise 标量广播 fast-path（绕 64 位 div/mod，4.4×）
- **P2b** — reduction 多级 partial 化（argmid kernel + 自适应阈值）
- **P-FFT-1/2** — fft batched PlanMany + cast_resize 合并（2D 24.5×）
- **P-A1** — argreduce partial 连续读对齐（argmax/argmin 64M +7%）
- **P-A3** — spmv/spmm musparse 描述符缓存（spmv 2000² 11×）
- **复数 reduction 分量并行** — c64 sum 1M 214ms→0.11ms（~1900×）

### Fixes

- 大块 buffer 立即 musaFree（跳过 pool/deferred 队列，修 bench_random 驱动 OOM）
- build.rs pkg-config 探测合并嵌套 if（clippy collapsible_if 门禁，pre-existing）
- bench Phase 7 复数项带宽口径（w=8 误计 16B/elem，虚高 2×）
- **cast 类型对 dispatch/validate 不一致** — f32/f64→i64 缺 GPU kernel 分支、
  complex→real 未拦截，真机 astype 触发 `unreachable!` panic；补 kernel/声明/
  mock/dispatch + validate 显式拒绝（`ca6611b`）

### Docs

- v0.3-alpha 计划（中英双语）+ ADR-003 草案
- SDK 3.1.0 限制集中汇总（sdk-3.1.0-limitations.md）
- repo.md 全量 benchmark 数据报告 + 更新
- Phase 2–8 各阶段状态更新 + 性能归因文档（P-A2/A4 证伪记录、复数带宽分析）
- benchmark/README.md — 大/中/小三档运行命令
- benchmark 与 README 的 dtype 参数迁移到字符串语法（`895a092`）

### Refactor

- 清理死代码与冗余实现（-245 行）：musaGetDevice/musaMemcpy2D/probe 等死 FFI、
  Layout::broadcast_to 死路径、parse_device 重复实现合并、redundant 比较简化、
  未用 import/死 helper/死赋值（`95358f3`）

### Bench

- Phase 7 — 复数 reduction + 多轴归约测量
- Phase 8 — 高级索引（mask/fancy）测量

## [v0.2.0-alpha] — 2026-08-04

完整算子面（elementwise/comparison/reduction/init/indexing）+ 性能优化 + 正确性修复。
详见 [docs/v0.2-alpha-release-note.md](docs/release/v0.2-alpha-release-note.md)。

## [v0.1.0-alpha] — 2026-07-28

核心运行时（Device/Dtype/Stream/Array/Buffer）+ 最小 add 算子。
详见 [docs/v0.1-alpha-release-note.md](docs/release/v0.1-alpha-release-note.md)。
