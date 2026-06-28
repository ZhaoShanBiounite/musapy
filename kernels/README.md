# kernels/

MUSA C kernel source files (`.mu` extension).

## Structure
kernels/
├── include/ # shared headers (common.h, elementwise.h, ...)
├── elementwise.mu # elementwise ops (add, sub, mul, div, sin, ...)
├── reduction.mu # reduction ops (sum, max, min, mean, ...)
├── indexing.mu # indexing ops (slice, gather, scatter, ...)
└── init.mu # init ops (zeros, ones, arange, ...)
See ADR L2-2 for kernel module contract.

## Build

Kernels are compiled by `rust/musapy-ops/build.rs` using `mcc`:
`.mu` → `.o` → `libmusapy_kernels.a`

Phase 6 (v0.1-alpha) will add the first kernel: `elementwise.mu` with `add`.
