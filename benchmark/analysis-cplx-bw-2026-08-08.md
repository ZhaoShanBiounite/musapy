# Benchmark 分析（2026-08-08）：复数 reduction 带宽高于实数（c64 vs f32）

## 现象

同规模下 c64 sum 每字节带宽是 f32 的 **~1.6-1.8×**（c64 只慢 1.1-1.26× 却搬 2× 数据）：

| 规模 | f32 sum | c64 sum | 延迟比 | 带宽比 |
|---|---|---|---|---|
| 1M | 0.091ms / 44 GB/s | 0.100ms / 80 GB/s | 1.10× | 1.82× |
| 16M | 0.347ms / 194 GB/s | 0.438ms / 306 GB/s | 1.26× | 1.58× |
| 64M | 1.173ms / 218 GB/s | 1.365ms / 375 GB/s | 1.16× | 1.72× |

（repo.md 里 f32 64M 219 与 c64 1M 0.107ms 是不同规模，直接比不对。）

## 对照实验排除干扰

同 16M 规模 elementwise vs reduction：

| 算子 | f32 | c64 | 带宽比 |
|---|---|---|---|
| elementwise add | 609 GB/s | 660 GB/s | 1.08×（无差距） |
| reduction sum | 194 GB/s | 306 GB/s | 1.58×（有差距） |

elementwise 无差距 → **排除**「c64 读 8B 天然带宽高」的通用访存粒度解释。

## 根因

reduction 内核**几乎纯访存**（每元素仅 1 次加法），吞吐由**访存指令 issue 速率**
（memory pipe 每周期可发的加载指令数）决定：

- f32 partial kernel：每线程 4 元素 = **4 条 LD.B32（4B/指令）**
- c64 partial kernel：每线程 4 元素 = **4 条 LD.B64（8B/指令）**
  （c64 元素是 8B struct `{float re, im}`，编译器对连续访问天然生成 8B 加载）

memory pipe 按指令数限速 → c64 每指令搬 2× 数据 → 带宽近 2×
（实测 1.6-1.8×，差额被 shuffle 归约/offset 计算占用的 issue slot 吃掉）。

**为何 elementwise 无此现象**：f32 elementwise 有 **float4 路径（LD.B128，
16B/指令）** 比 c64 的 LD.B64 更宽，且数学运算占用 issue slot——f32 不落后。

## 与其他发现的呼应

- P-A4 探针：f64 elementwise items=1 697 > f32 636 GB/s——同源（纯访存时
  8B 加载 > 4B）。
- P2 注释：f32 reduction 无法用显式 float4 提速——「float4+shuffle 组合在
  本编译器病态（实测 47× 变慢）」，是编译器限制。

## 含义

f32 reduction 带宽受限是**编译器/硬件访存宽度限制**（LD.B32 4B/指令），
非应用层可解（float4+shuffle 病态）；c64 的 8B 元素天然占便宜。
若 SDK/编译器升级后 float4+shuffle 可用，f32 reduction 预期可翻倍。
