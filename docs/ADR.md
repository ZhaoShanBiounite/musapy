# musapy Architecture Decision Records (ADR)

> **Status**: Draft
> **Last Updated**: 2025-01-15
> **Scope**: musapy v1.0 design (Python + Rust + MUSA scientific computing library)

This document records all architecture decisions for musapy, organized into 5 layers.
Each decision has a stable ID (`L<layer>-<number>`) for reference in code, issues, and future ADRs.

---

## Table of Contents

- [Layer 0: Positioning](#layer-0-positioning)
- [Layer 1: Core Abstractions](#layer-1-core-abstractions)
- [Layer 2: Module Contracts](#layer-2-module-contracts)
- [Layer 3.1: Error Model](#layer-31-error-model)
- [Layer 3.2: Memory Lifecycle](#layer-32-memory-lifecycle)
- [Layer 3.3: Interoperability](#layer-33-interoperability)
- [Layer 3.4: Observability](#layer-34-observability)
- [Layer 4: Evolution Strategy](#layer-4-evolution-strategy)
- [Appendix: Decision Index](#appendix-decision-index)

---

## Layer 0: Positioning

### L0-1: Primary Audience

**Decision**: Scientific computing users first (C), HPC users as evolution target (A).

**Rationale**: MUSA ecosystem maturity is insufficient for ML users (B) today. Scientific users
benefit most from a CuPy-equivalent on MUSA. HPC users come later as the ecosystem matures.

**Implications**:
- "Explicit device" is a feature, not a burden (HPC mindset)
- CPU fallback via separate `musapy-cpu` crate (opt-in), not main library
- PyTorch compatibility (`__torch_function__`, autograd) deferred to v3+

### L0-2: Primary Scenario

**Decision**: Single-GPU acceleration first, large-scale offline computation as evolution target.

**Implications**:
- v1 does NOT include distributed (MCCL) or ShardedArray
- Distributed is v2+ scope
- Large-scale offline (Dask/Spark-GPU style) is far future

### L0-3: Ecosystem Position

**Decision**: Independent usable library. Not a PyTorch backend. Not prioritizing PyTorch compat.

**Implications**:
- DLPack interop done, but PyTorch cross-validation deferred
- `__torch_function__` and autograd integration are v2+/v3+
- musapy is a product, not infrastructure for another framework

### L0-4: Technology Stack

**Decision**: Python + Rust + MUSA. Locked, non-negotiable.

**Rationale**:
- Python: user-facing API, NumPy/SciPy-compatible surface
- Rust: memory safety, PyO3 FFI, error handling via `Result`
- MUSA: Moore Threads GPU stack (MUBLAS/MUDNN/MUSPARSE/MURAND/MUSOLVER/MUFFT/MCCL)

### L0-5: Library Alias

**Decision**: Recommend `import musapy as ms`. Do not force. Do not encourage `from musapy import *`.

**Rationale**: `ms` is short, mnemonic, non-conflicting. Public API accessed via `ms.` prefix
avoids namespace pollution. Not forcing alias allows user preference.

### L0-6: Device Resolution — 5-Level Priority Chain

**Decision**: Device parameter always exists in API, but its value resolves through a 5-level chain:

| Priority | Source | Description |
|---|---|---|
| 1 | Function call `device=` arg | Highest, always wins |
| 2 | `with ms.device(...)` context | Overrides global |
| 3 | Input Array's device (ufunc-style) | `a + b` follows `a` |
| 4 | Global default device (`ms.set_default_device`) | Process-wide, thread-local |
| 5 | Auto-probe at startup | Prefer MUSA over CPU if available |

**Key**: Every level is overridable by higher levels. Every resolution is traceable.

### L0-7: Dtype Resolution — Symmetric to Device

**Decision**: Dtype follows the same 5-level resolution chain:

| Priority | Source |
|---|---|
| 1 | `dtype=` arg |
| 2 | `with ms.dtype(...)` context |
| 3 | Input Array's dtype (type promotion result) |
| 4 | Global default dtype (thread-local) |
| 5 | Startup default `ms.float32` |

**Note**: Dtype auto-probe is meaningless (unlike device hardware detection), so level 5 is fixed
`float32`. `DeviceNotConfigured` only fires for device, dtype always has fallback.

### L0-8: Feedback Principle

**Decision**: Every device/dtype resolution must produce a traceable `DeviceResolution` /
`DtypeResolution` record attached to the Array, including source level + source location.

**Example**:
```python
>>> a = ms.array([1,2,3])
>>> a.device
Device(musa:0)  # resolved from: global_default (musa:0), set via mp.set_default_device() at <stdin>:2
```

**Rationale**: Not a perf optimization — a correctness guarantee. Distributed debugging relies on
this to locate "why did data end up on the wrong device".

### L0-9: First-Creation Requires Explicit Device

**Decision**: If `ms.set_default_device()` was never called, the first `ms.array()` raises
`DeviceNotConfigured` instead of silently using auto-probe.

**Rationale**: For scientific users, data on CPU vs MUSA is night-and-day different. Silent
wrong choice invalidates result comparison.

### L0-10: Cross-Device Operations — Strict Policy

**Decision**: Cross-device operations (e.g., `a` on musa:0 + `b` on musa:1) raise
`DeviceMismatch`. User must explicitly `.to(device)`.

**Rationale**: Implicit cross-device migration is a perf trap. Explicit `.to()` makes data
movement visible.

### L0-11: Default Device Model — Hybrid (thread-local + thread-safe runtime)

**Decision**:
- Default device/dtype: **thread-local stack** (per-thread isolation, zero-lock)
- Runtime infrastructure (handle tables, memory pools, stream pools): **thread-safe**
  (RwLock/DashMap/AtomicPtr)
- New threads inherit parent thread's default at `start()` time (value snapshot, then decoupled)
- No broadcast API. Workers read config themselves at startup.

**Rationale**: HPC workload analysis shows thread-local wins 6 of 8 dimensions vs global-shared.
See ADR-L0-11 detailed comparison in design notes.

---

## Layer 1: Core Abstractions

### L1-1: Device Identifier

**Decision**: Both string (`"musa:0"`) and `Device` object supported.

```python
ms.array([1,2,3], device="musa:0")           # string
ms.array([1,2,3], device=ms.Device.musa(0))  # object
```

### L1-2: Unavailable Device — Fail at Startup

**Decision**: `ms.set_default_device("musa:5")` on a machine with only 1 GPU raises
`DeviceUnavailable` immediately, not deferred to first op.

### L1-3: Device Capability Query

**Decision**: Expose `device_count`, `arch` (compute capability), `total_memory` (per-device),
`total_memory_all_devices` (aggregate).

```python
ms.device_summary()
# musa:0 — MTT S4000, arch=mp_22, 47.9 GB VRAM, 56 CUs
# musa:1 — MTT S4000, arch=mp_22, 47.9 GB VRAM, 56 CUs
```

### L1-4: Phase 1 Dtype Set

**Decision**: 15 dtypes with extension slots reserved.

```
bool, int8, int16, int32, int64,
uint8, uint16, uint32, uint64,
float16, float32, float64, bfloat16,
complex64, complex128
```

**Rationale**: Scientific users need complex (FFT, signal processing) and bfloat16 (numerical
experiment comparison). Missing them forces NumPy fallback, fragmenting experience.

### L1-5: Type Promotion — JAX-Style Type-Based

**Decision**: Use JAX's type-based promotion table. No value-based inference (avoids NumPy's
`int8 + 1 → int64` trap).

### L1-6: bfloat16 on CPU

**Decision**: CPU bf16 operations auto-promote to f32 for computation, then round back to bf16.

**Rationale**: CPU has no bf16 hardware support. Documented behavior, not implicit.

### L1-7: Default Stream

**Decision**: Each device has one default stream, owned by runtime. `ms.array(...)` without
explicit stream binds to the device's default stream. Null stream is NOT exposed.

**Rationale**: Null stream's implicit-sync semantics is CUDA legacy baggage. New library should
not inherit. Default stream avoids forcing `with stream:` for every op.

### L1-8: `out=` Parameter Stream Semantics

**Decision**: Op executes on `out`'s stream. Runtime auto-inserts wait for input streams.

```python
with ms.stream(s1): a = ms.array(...)
with ms.stream(s2):
    b = ms.array(...)
    c = ms.empty(...)
    ms.add(a, b, out=c)   # executes on s2, auto-waits s1 for a
```

**Rationale**: Matches "out is result container" intuition. Friendlier than error.
Debug mode logs all auto-inserted waits (feedback principle).

### L1-9: Stream Priority

**Decision**: Exposed via `ms.Stream(device, priority=...)`. Aligned with MUSA stream priorities.

### L1-10: Buffer Read/Write Reference Separation

**Decision**:
- `Arc<Buffer>`: writable, unique ownership semantics
- `BufferRef(Arc<Buffer>)`: read-only shared view
- Op inputs auto-downgrade to `BufferRef`; outputs are new `Buffer`

**Rationale**: Enables `__restrict__` in kernels (compiler can assume no aliasing). Compile-time
aliasing detection (same `BufferRef` cannot be both input and `out`).

### L1-11: 0-dim Array

**Decision**: No special scalar path. `shape=[]` is just 0-dim. MUSA runtime auto-optimizes.
BUT `.item()` / `__float__` / `__int__` explicitly trigger `stream.synchronize` + D2H copy.

```python
a = ms.array(3.14, device="musa:0")  # 0-dim on GPU
a + 1                                  # OK, no sync, result is 0-dim GPU Array
float(a)                               # triggers sync + D2H
```

### L1-12: Execution Model — Eager with Lazy Hook

**Decision**: Eager execution primary. Op functions internally use `OpBuilder` which separates
parameter parsing (once) from kernel launch (replayable). This preserves a lazy hook for future
MUSA Graphs capture without API breakage.

**Constraint**: All op functions must be **capture-safe** — no host-side mutable state reads
during execution phase.

### L1-13: Device Policy — Strict Default

**Decision**: Default policy is `strict`: cross-device ops raise `DeviceMismatch`.

"Musa > CPU" hierarchy is ONLY used for `auto` default device probe preference. Does NOT affect
op behavior.

### L1-14: GPU Precision Alignment (Dtype Policy)

**Decision**: Two-tier rule (default behavior, no opt-in needed):

| Scenario | Rule |
|---|---|
| All-GPU operation (all inputs on MUSA) | Result dtype = **narrowest** input dtype (f16 > bf16 > f32 > f64, performance priority) |
| Contains CPU operation (any input on CPU) | Use JAX standard promotion table (correctness priority) |

**Same-width conflict rule**: bf16 + f16 (both 16-bit) → JAX promotion → f32 (avoid precision loss).

**Extension table** (official musapy type promotion spec):

| Input combo (all-GPU) | Result | Reason |
|---|---|---|
| f16 + f32 | f16 | narrow priority |
| bf16 + f32 | bf16 | narrow priority |
| f16 + bf16 | f32 | same-width conflict → JAX |
| f32 + f64 | f32 | narrow priority |
| f32 + i32 | f32 | int→float (JAX), GPU narrow → f32 |
| i32 + i64 | i32 | int narrow priority |
| i32 + u32 | i64 | JAX (signed+unsigned may overflow) |
| f32 + complex64 | complex64 | complex narrow priority |
| complex64 + complex128 | complex64 | narrow priority |
| bool + f32 | f32 | bool→float |

### L1-15: Green Context

**Decision**: v1 does NOT use Green Context. Thread-local default + thread-safe runtime is
sufficient. Green Context deferred to v2+.

**Rationale**: Green Context documentation is thin. Mature thread-local + thread-safe solution
first. Green Context is an optimization, not a requirement.

### L1-16: OpBuilder and MUSA Graphs

**Decision**: OpBuilder's lazy hook targets MUSA Graphs API (not self-built DAG).

**Implication**: All ops must be capture-safe (parameter parsing separable from kernel launch).

---

## Layer 2: Module Contracts

### L2-1: Build System

**Decision**:
- `maturin` + Cargo workspace + `mcc` compiles `.mu` → `.o` → `libmusapy_kernels.a`
- ABI version embedded in symbol names: `musapy_mul_f32_v1`
- Runtime checks kernel ABI at startup
- MUSA SDK detection: `MUSA_HOME` env + pkg-config dual probe
- MUSA Runtime version (from musart_version.h) vs runtime ABI version compatibility matrix check

**Forbidden**:
- Runtime logic in build scripts
- Hardcoded MUSA paths

### L2-2: MUSA Kernels (`kernels/*.mu`)

**Responsibilities**: Pure parallel compute kernels. Thread grid logic. Branchless math.
Device-side memory access (read-only inputs, write-only outputs).

**Allowed deps**: `musa_runtime.h`, `include/` headers, MUSA intrinsics.

**Forbidden**:
- Memory alloc/free (`malloc`, `musaMalloc`)
- Host-side code (`printf`, file I/O)
- Error returns (kernels return `void`)
- Scheduling logic (grid/block size decisions)
- Runtime type branching (`if dtype==...` — must template-instantiate)
- Cross-device operations

**Interface contract**: Pure C, stateless: `extern "C" void musapy_<op>_<dtype>_v<abi>(...)`.
All pointers `__restrict__` (guaranteed by ops layer alias detection).

### L2-3: Core Runtime (`rust/musapy-core`)

**Responsibilities**:
- Data structure definitions and invariants (Array / Buffer / BufferRef / Device / Dtype /
  Stream / Layout / DeviceResolution)
- Memory lifecycle (RAII + stream-ordered dealloc)
- Thread-safe global infrastructure (device table, memory pool, stream pool, MUBLAS handle table)
- Thread-local default device/dtype stacks
- DLPack interop

**Allowed deps**: MUSA runtime API only (NOT compute libraries like MUBLAS/MUDNN). Standard
Rust crates.

**Forbidden**:
- Any operator implementation
- Calling MUBLAS/MUDNN/MCCL
- Operator dispatch
- Modifying Array's device/dtype fields (read-only use)
- Direct Python object/GIL management

**Thread safety layering**:
```
Global read-only (immutable after startup):
  Device table, capabilities, ABI version → no lock needed

Global mutable (thread-safe):
  Memory pool → RwLock<MemoryPool>
  Stream pool → DashMap<DeviceId, Arc<Stream>>
  MUBLAS handle table → thread_local<Handle> (handle not thread-safe)

Thread-local:
  Default device stack → RefCell<Vec<Device>>
  Default dtype stack → RefCell<Vec<Dtype>>
  Current stream stack → RefCell<Vec<Arc<Stream>>>
```

### L2-4: Ops Layer — Capture-Safe Constraint

**Decision**: All op functions must be capture-safe:
- Parameter parsing (shape/dtype/device checks) executes once
- Kernel launch is replayable
- No host-side mutable state reads during execution phase
- Parameter parsing and kernel launch are separated in `OpBuilder`

### L2-5: Ops Layer — Alias Detection

**Decision**: Same `BufferRef` cannot be both op input and `out` parameter. Violation raises
`AliasDetected` error. No auto-copy.

**Rationale**: Enables `__restrict__` in kernels. Compile-time guarantee via Buffer/BufferRef
type separation.

### L2-6: PyO3 Binding — Stream-Aware DLPack Export

**Decision**: `__dlpack__(stream)` export:
1. Record current array's pending write event
2. If consumer passed stream, make it wait our event
3. Capsule holds event reference (prevents buffer release before event completes)

### L2-7: Python Frontend — Context Composition

**Decision**: `ms.device()` / `ms.dtype()` / `ms.stream()` context managers are symmetric and
composable. Support arbitrary nesting and tuple shorthand:

```python
with ms.device("musa:0"), ms.stream(s1), ms.dtype(ms.float16):
    ...
```

### L2-8: Python Frontend — Import Style

**Decision**: Do not encourage `from musapy import *`. Do not force `import musapy as ms`.
Public API accessed via `musapy.` prefix. Documentation examples use `ms` alias.

---

## Layer 3.1: Error Model

### L3-1: Two-Layer Detection

**Decision**:
- **Launch errors** (invalid params, bad handle): check `musaGetLastError` immediately after
  op queue. Report at op call site.
- **Execution errors** (out-of-bounds, NaN): deferred to `stream.synchronize()`. Reported with
  op context from stream's pending queue.

### L3-2: OpContext Attribution

**Decision**: Each op queued records an `OpContext` to the stream's pending queue:

```rust
pub struct OpContext {
    op_name: &'static str,        // "matmul"
    input_shapes: Vec<Shape>,
    input_devices: Vec<Device>,
    input_dtypes: Vec<Dtype>,
    output_shape: Shape,
    stream_id: u64,
    python_frame: Option<PythonFrame>,  // debug mode only
    timestamp: Instant,
}
```

On synchronize error, find the last uncompleted op (most likely root cause) and attach to
error message.

### L3-3: Poison Recovery

**Decision**:
- Op execution failure marks stream `poisoned: AtomicBool`
- Poisoned stream: all subsequent ops immediately return `PoisonedStream` (no queueing)
- v1 provides `stream.reset()` (marked `@experimental`): destroys stream + invalidates all
  buffers owned by that stream. Does NOT guarantee context consistency. Production should
  restart process.
- `ms.reset_device()` deferred to v2+

### L3-4: Capture Mode Errors

**Decision**:
- Parameter errors + launch errors: reported immediately at capture time (musaGraphAddNode
  validates)
- Execution errors: reported at `graph.replay()`, attributed to graph node, mapped back to
  OpContext

### L3-5: Exception Hierarchy Depth

**Decision**: Shallow inheritance, two levels:

```
MusapyError
├── DeviceError
├── DtypeError
├── ShapeError
├── MemoryError
├── StreamError
├── KernelError
└── InteropError
```

### L3-6: Python Built-in Exception Inheritance

**Decision**: Partial inheritance from Python built-ins:

```
MusapyError(Exception)
├── DeviceError(MusapyError, RuntimeError)
├── DtypeError(MusapyError, TypeError)
├── ShapeError(MusapyError, ValueError)
├── MemoryError(MusapyError)                    # see L3-7
├── StreamError(MusapyError, RuntimeError)
├── KernelError(MusapyError, RuntimeError)
└── InteropError(MusapyError, RuntimeError)
```

**Rationale**: Scientific users migrating from NumPy have `except ValueError` muscle memory.
Partial inheritance maintains compatibility.

### L3-7: OutOfMemoryError — No Built-in Inheritance

**Decision**: `OutOfMemoryError(MusapyError)` does NOT inherit Python's built-in `MemoryError`.

**Rationale**: GPU VRAM exhaustion is semantically different from Python heap memory exhaustion.
Mixing them misleads users.

**Full exception hierarchy**:

```
MusapyError(Exception)
├── DeviceError(MusapyError, RuntimeError)
│   ├── DeviceNotConfiguredError
│   ├── DeviceMismatchError
│   └── DeviceUnavailableError
├── DtypeError(MusapyError, TypeError)
│   └── UnsupportedDtypeError
├── ShapeError(MusapyError, ValueError)
│   └── ShapeMismatchError
├── MemoryError(MusapyError)
│   ├── OutOfMemoryError
│   └── AliasDetectedError
├── StreamError(MusapyError, RuntimeError)
│   ├── PoisonedStreamError
│   └── SyncCycleError
├── KernelError(MusapyError, RuntimeError)
│   ├── LaunchFailedError
│   └── KernelFailedError
└── InteropError(MusapyError, RuntimeError)
    ├── DlpackExportError
    └── UnsupportedProtocolError
```

---

## Layer 3.2: Memory Lifecycle

### L3-8: Memory Pool — 3-Layer Structure

**Decision**:

| Layer | Responsibility |
|---|---|
| L1 MUSA runtime | `musaMallocAsync` / `musaFreeAsync`, stream-ordered |
| L2 musapy BufferPool | per-device pool, size-class bucketed reuse |
| L3 Buffer (user-facing) | RAII handle, Drop returns to pool (not immediate free) |

**GC strategy**: B (periodic + LRU). Default: every 60s or explicit `ms.gc(device)`, free
buffers unused >5min.

**User-configurable**: `ms.set_memory_policy("aggressive" | "lazy" | "manual")`

**Implementation status (Phase C-lite, 2025-07)**: L2 BufferPool implemented
(`buffer_pool.rs`), compiled only on the default path
(`#[cfg(not(feature = "stream-ordered"))]`). Design parameters:
- SizeClass = round_up_pow2(size), minimum 512 bytes
- Per-device cache cap 512 MB; overflow falls back to deferred-free (L3-11)
- Cross-stream reuse waits on stored event (safety guarantee)
- Reuse requires `actual_size >= requested_size` (same size-class may hold smaller entries)
- GC policy (LRU eviction, `ms.gc()`) not yet implemented; only capacity cap enforced

### L3-9: Stream-Ordered Dealloc — Conditional Implementation (feature gate)

**Decision**: v1 supports both paths simultaneously, selected via Cargo feature gate
+ runtime probe:

| Build mode | feature | alloc/free API | SDK support |
|---|---|---|---|
| Default | (none) | musaMalloc / musaFree + deferred-free queue | 3.x / 4.x / 5.x |
| stream-ordered | `stream-ordered` | musaMallocAsync / musaFreeAsync | 5.x+ |

**Rationale**: MUSA Runtime 3.x/4.x libmusart.so does not contain musaMallocAsync/
musaFreeAsync symbols (verified on 3.1.0/3.3.5/4.3.7). MUSA SDK 5.1.0 Release Notes
explicitly states "added support for Stream Ordered Memory Allocator API" (CUDA 12.8
equivalent), but 5.x is currently restricted release. To keep a single codebase
compatible with all versions, feature gate controls async API link declarations,
runtime probe acts as double safety.

**Verified version matrix (2025-01)**:

| MUSA Runtime | musaMallocAsync | musaFreeAsync | musaMalloc/Free |
|---|---|---|---|
| 3.1.0 | header declares, .so no symbol | header declares, .so no symbol | ✅ available |
| 3.3.5 | header declares, .so no symbol | header declares, .so no symbol | ✅ available |
| 4.3.7 | C++ inline wrapper (forwards to musaMallocFromPoolAsync) | declared only, no impl | ✅ available |
| 5.1.0 | ✅ full | ✅ full | ✅ available |

**Future**: Once 5.x is publicly available, change `stream-ordered` to default feature,
or remove the feature gate entirely, unifying on stream-ordered path.

### L3-10: Dealloc Stream Selection Strategy

**Decision**: Strategy **b** (last-used stream). Buffer's `dealloc_stream` is mutable, updated
on cross-stream use.

**Optimization**: `read_events` only stores events not yet waited-on by `dealloc_stream`. Pop
after wait. Vec typically 0-1 elements.

**Phase C-lite same-stream optimization (2025-07)**: Buffer gains
`last_write_stream_id: AtomicU64`. When read/write ops share the same stream
(the common single-stream case):
- `wait_last_write_on`: same stream → return Ok immediately (skip musaStreamWaitEvent)
- `record_write`: consecutive same-stream writes skip Event::new/Record (implicit ordering)
- `record_read`: same-stream read skips event creation
Measured reduction: ~6 driver calls/op, ~39% latency improvement on small arrays.

```rust
impl Drop for Buffer {
    fn drop(&mut self) {
        let dealloc = self.dealloc_stream.lock().unwrap().clone();
        for ev in self.read_events.lock().unwrap().drain(..) {
            dealloc.wait_event(&ev);
        }
        if let Some(ev) = self.last_write_event.lock().unwrap().take() {
            dealloc.wait_event(&ev);
        }
        unsafe { musaFreeAsync(self.ptr, dealloc.raw()); }
    }
}
```

### L3-11: Deferred-Free — Default Path

**Decision**: deferred-free is the default build path, compatible with all SDK versions
(3.x/4.x/5.x). stream-ordered (L3-9) is an optional feature, enabled on 5.x environments.

**Workflow**:
1. `Buffer::drop` does not free immediately; instead enqueues `(ptr, events)` to a
global deferred-free queue
2. Before enqueuing, wait all read/write events on `dealloc_stream` (strategy b guarantee)
3. After `Stream::synchronize` succeeds, batch reclaim: call `musaFree(ptr)` for all
buffers in the queue

**Safety guarantees**:
- synchronize guarantees all ops on the stream are complete
- events are waited before enqueuing, so after synchronize they are certainly complete
- Therefore when reclaiming, the buffer is certainly not in use by any stream

**Relationship with L3-9**: L3-11 is the fallback when L3-9 is unavailable, and also the
current default path. When `stream-ordered` feature is enabled, Buffer takes the L3-9 path,
and the deferred-free queue is no longer used (code is preserved, auto-restored when
feature is off).

**Relationship with L3-8 BufferPool (Phase C-lite)**: On the default path, `Buffer::drop`
first attempts to return the buffer to BufferPool for reuse; only when the pool is full
(>512 MB/device) does it fall back to the deferred-free queue.
i.e.: BufferPool is the hot path, deferred-free is the cold-path safety net.

**Capability probe**: Startup probes `musaDeviceGetAttribute(MUSA_DEV_ATTR_MEMORY_POOLS_SUPPORTED)`.
Even when compiled with `stream-ordered` feature, probe acts as double safety — if the
runtime doesn't support it, fallback to deferred-free.

### L3-12: DLPack Lifecycle

**Decision**: DLPack capsule holds `Arc<Buffer>`. Reference count guarantees buffer not recycled
before capsule release. Capsule release → Arc decrement → potential Drop → normal stream-ordered
free flow.

### L3-13: Poison Recovery — Conservative Strategy

**Decision**: `stream.reset()` destroys ALL buffers owned by that stream, marks invalid.
Subsequent access raises `BufferInvalidated`.

**Rationale**: MUSA errors are sticky. Cannot precisely determine which buffers are affected.
Conservative strategy avoids use-after-poison.

### L3-14: Graph Placeholder Semantics

**Decision**:
- During capture, Arrays marked `is_graph_placeholder: true`
- Normal ops receiving placeholder raise `GraphNotReplayed` (cannot use outside capture)
- After `graph.replay()`, placeholder buffers are populated, Array converts to normal
- v1 replay is synchronous

### L3-15: Minimal Verification Test

**Decision**: Before v1 implementation, a colleague with MUSA hardware runs a minimal
cross-stream alloc/use/free test to verify stream-ordered dealloc works on real MUSA hardware.

**Status (2025-01)**: Verified on MUSA Runtime 3.1.0 / 3.3.5 / 4.3.7 that stream-ordered
API is unavailable (musaFreeAsync has no implementation), falling back to deferred-free.
5.1.0 environment pending verification.

**Test scope**: Once 5.x is available, run the stream-ordered verification test (see
v0.1-alpha-plan section 2.1). If it passes, the `stream-ordered` feature can be enabled
as the recommended build mode.
---

## Layer 3.3: Interoperability

### L3-16: DLPack MUSA Device Type

**Decision**: Define custom `kDLMUSA` (using DLPack reserved range, e.g., value 100). Submit
to DLPack upstream for official enum value when stable.

### L3-17: `__array_ufunc__` Cross-Device

**Decision**: In strict policy, `np.add(musapy_array_on_gpu, numpy_array_on_cpu)` raises
`DeviceMismatch` with explicit fix message:

```
ms.DeviceMismatch: np.add() received inputs on different devices
  musapy Array on musa:0
  numpy.ndarray on cpu
  fix: convert numpy array to musapy first, e.g. np.add(a, ms.array(b, device=a.device))
  or:  convert musapy array to numpy first, e.g. np.add(a.cpu().numpy(), b)
```

### L3-18: `__array_function__` v1 Scope

**Decision**: v1 supports high-frequency functions only: `concatenate`, `stack`, `split`,
`zeros_like`, `ones_like`, `where`. Others fallback to NumPy (triggers `.cpu()` sync).

Extension space reserved for future additions.

### L3-19: `__array__` Implicit Sync

**Decision**: `np.array(musapy_array)` triggers `stream.synchronize` + D2H copy. No warning.
Documented as sync operation.

### L3-20: DLPack v1 Implementation

**Decision**: v1 implements DLPack protocol with round-trip validation (musapy export →
musapy import). Cross-library validation (torch_musa) deferred.

### L3-21: PyTorch Interop

**Decision**: v1 does NOT validate cross-library interop with PyTorch/torch_musa.
`__torch_function__` and autograd integration deferred to v2+/v3+.

### L3-22: CuPy Interop

**Decision**: v1 does NOT do dedicated CuPy interop. CuPy is CUDA-only, physically incompatible
with MUSA GPUs. Document: "to mix with CuPy, use explicit `.cpu()` + `cp.asarray()`".

### L3-23: Cross-Device/Framework Interop

**Decision**: v1 focuses on pure Moore Threads environment. No cross-device or cross-framework
interop validation.

### L3-24: Interop Error Handling

**Decision**:

| Error scenario | Reporter | Error type |
|---|---|---|
| DLPack export with invalidated buffer | musapy | `InteropError.DlpackExport` |
| DLPack import with bad capsule | musapy | `InteropError` |
| Consumer accesses buffer before event completes | consumer | consumer's own error |
| `__array_ufunc__` cross-device | musapy | `DeviceMismatch` |
| `__array__` with poisoned stream | musapy | `PoisonedStream` |

Debug mode: assert Arc strong count == 1 in capsule deleter, else panic.

---

## Layer 3.4: Observability

### L3-25: Profiling — No Self-Built

**Decision**: musapy does NOT build its own profiler. Use Moore Threads' `msys profile`
(Moore Perf System) and Moore Perf Compute (MCU) for all profiling needs.

**Rationale**: Moore Perf System already provides kernel timeline, stream swimlanes, API
tracing, GPU metrics. Moore Perf Compute provides Roofline, kernel-level analysis. musapy should
not duplicate.

**Documentation**: guide users to run `msys profile -t musa -o report.msys-rep python script.py`.

**OpContext** (from L3-2) is still recorded for error attribution, NOT for profiling.

### L3-26: Debug Mode — Runtime Flag

**Decision**: Single binary, runtime flag. `ms.set_debug(True)` or `MUSAPY_DEBUG=1` env var or
`with ms.debug():` context.

**Debug mode enables**:
- OpContext records `python_frame`
- Sync DAG full cycle detection (DFS)
- Buffer alias detection + detailed dump
- Arc count assert (L3-24)
- Freed buffers filled with `0xDEADBEEF` (use-after-free visualization)
- Op parameter full dump to log

**Implementation**: Rust `if debug` branches, compiler eliminates release path. Zero overhead
when debug off.

### L3-27: Array Naming

**Decision**: `name` stored at Array layer (not Buffer). `Array.name: Option<String>`.

**Rationale**:
- Same buffer may have multiple views (slice/transpose), each with different name
- Buffer is hot-path data structure, String field hurts cache locality
- Array count << Buffer count, overhead negligible

```python
a = ms.array(..., device="musa:0", name="weights.layer1")
# or
a.name = "weights.layer1"
```

### L3-28: Memory/Stream State Query

**Decision**:
- `ms.memory_summary(device)` / `ms.stream_summary()` / `ms.device_summary()`: use atomic
  counters (zero-lock), suitable for frequent monitoring
- `ms.memory_detail(device)`: traverses BufferPool, explicitly documented as having overhead

**Atomic counters maintained**:
- `allocated_bytes`, `allocated_buffers`
- `cached_bytes`, `cached_buffers`
- `peak_bytes`, `peak_timestamp`

---

## Layer 4: Evolution Strategy

### L4-1: v1 Ops Scope

**Decision**: v1 implements:

| Op category | v1 | Source |
|---|---|---|
| elementwise (add/sub/mul/div/sin/cos/exp/log/pow/abs/sign/clamp) | ✅ | custom `.mu` kernels |
| reduction (sum/max/min/mean/argmax/argmin/cumsum/prod) | ✅ | custom `.mu` kernels |
| init (zeros/ones/arange/linspace/fill/eye) | ✅ | custom `.mu` kernels |
| linalg (matmul/lu/qr/svd/solve) | ✅ | muBLAS + muSOLVER |
| random (rand/randn/uniform/normal/bernoulli) | ✅ | muRAND |
| fft (fft/fftn/ifft/rfft) | ✅ | muFFT |
| sparse (csr_matrix/coo_matrix/spmv/spmm) | ✅ | muSPARSE |
| indexing (slice/gather/scatter/transpose/permute/flip) | ✅ | custom `.mu` kernels |
| broadcast | ✅ | via strides=0 in elementwise |
| comparison (==/!=/</>/<=/>=/argmax) | ✅ | custom `.mu` kernels |

**v1 does NOT implement**:
- nn (muDNN: conv/pool/activation/batch_norm/softmax) — v2+
- distributed (MCCL: all_reduce/all_gather/broadcast/send/recv) — v2+

### L4-2: v1 Excluded Items

| Item | Deferred to |
|---|---|
| Distributed (MCCL) | v2 |
| MUSA Graphs capture implementation | v2 (v1 keeps OpBuilder hook only) |
| PyTorch interop validation | v2+ |
| `__torch_function__` / autograd | v2+/v3+ |
| Green Context | v2+ |
| `ms.reset_device()` | v2+ |
| Kernel fusion / JIT | v3+ |
| Autotuning | v3+ |
| CPU fallback crate (`musapy-cpu`) | v2+ (if adoption needs) |
| ShardedArray | v2+ |
| StreamedArray | v2+ |

### L4-3: Backward Compatibility Policy

**Decision**: SemVer with musapy-specific clarifications:

| Change type | Policy |
|---|---|
| Python public API signature | minor: no break; major: may break |
| Rust crate public API | same, Cargo SemVer compatible |
| Kernel ABI (symbol names) | minor: no break (new symbols use `_v2` suffix, old preserved); major: may break |
| Default behavior change (device/dtype policy) | minor: no default change, add opt-in; major: may change default |
| Experimental API | annotated `@experimental`, minor may break, release notes must document |
| Error message format | no compatibility guarantee (users should not parse error messages) |

**Experimental API graduation flow**:
- experimental → stable candidate (1 minor version) → stable (next minor)
- During candidate phase: collect feedback, refine API
- At stable: API signature frozen, standard compat policy applies

### L4-4: Deprecation Flow

**Decision**:

| Stage | Behavior |
|---|---|
| 1. Mark deprecated (vX.Y) | API still works, emits `DeprecationWarning`, docs show replacement |
| 2. Retain 1 major cycle (v(X+1) still works) | continues warning, not removed |
| 3. Remove (v(X+2)) | actually deleted |

Use Python standard library `DeprecationWarning` (not custom). Users silence via standard
`warnings.filterwarnings`.

### L4-5: Pre-Release Sequence

**Decision**:

| Version | Scope |
|---|---|
| v0.1-alpha | Core runtime (Device/Dtype/Stream/Array/Buffer) |
| v0.2-alpha | Basic ops (elementwise/reduction/init/indexing/broadcast/comparison) |
| v0.3-alpha | Math library ops (linalg/random/fft/sparse) |
| v0.4-beta | Interop + error model + observability complete |
| v1.0-rc1 | Full v1 scope, public testing |
| v1.0 | Stable release |

### L4-6: MCCL + Graphs

**Decision**: Both MCCL distributed and MUSA Graphs capture implementation deferred to v2.

**Rationale**:
- MCCL requires multi-GPU test environment (≥2 MUSA GPUs)
- MUSA Graphs API maturity needs verification on real hardware
- v1.0 should land ASAP for user adoption
- Both are relatively independent modules, can be added as v1.1 or v2.0 minor/major

---

## Appendix: Decision Index

### By Layer

| Layer | Decisions | Range |
|---|---|---|
| Layer 0 | 11 | L0-1 to L0-11 |
| Layer 1 | 16 | L1-1 to L1-16 |
| Layer 2 | 8 | L2-1 to L2-8 |
| Layer 3.1 | 7 | L3-1 to L3-7 |
| Layer 3.2 | 8 | L3-8 to L3-15 |
| Layer 3.3 | 9 | L3-16 to L3-24 |
| Layer 3.4 | 4 | L3-25 to L3-28 |
| Layer 4 | 6 | L4-1 to L4-6 |
| **Total** | **69** | |

### By Status

| Status | Count | IDs |
|---|---|---|
| Accepted (final) | 69 | All |
| Experimental | 1 | L3-3 (`stream.reset()`) |
| Deferred to v2+ | 11 | L1-15, L3-3 (reset_device part), L4-2 items |

### Key Cross-References

- **Device policy**: L0-6, L0-9, L0-10, L1-13, L3-17
- **Dtype policy**: L0-7, L1-4, L1-5, L1-14
- **Stream model**: L1-7, L1-8, L1-9, L3-1, L3-2, L3-10
- **Memory lifecycle**: L1-10, L2-3, L3-8 to L3-15
- **Error model**: L3-1 to L3-7, L3-24
- **Interop**: L2-6, L3-12, L3-16 to L3-24
- **Capture-safety**: L1-12, L1-16, L2-4, L3-4, L3-14

---

## Change Log

| Date | Change | ADR IDs affected |
|---|---|---|
| 2025-01-15 | Initial draft, all 69 decisions | All |

---

## Notes on Using This ADR

1. **Referencing decisions**: use stable IDs like `ADR-L1-10` or `L1-10` in code comments,
   issues, PRs.
2. **Proposing changes**: open a new ADR (e.g., ADR-002) that supersedes specific decisions.
   Do NOT edit this file directly for changes.
3. **Experimental items**: track graduation in separate `EXPERIMENTAL.md`.
4. **Implementation tracking**: each decision should have a tracking issue in the musapy repo.

---

*End of ADR*
