#!/usr/bin/env bash
# check_musax_symbols.sh —— v0.3 Phase 1 P1.1 运行期符号审计
#
# 用 nm -D 核对 MUSA-X 五个数学库 .so 是否导出 musapy v0.3 计划用到的全部符号
# （Phase 1 生命周期 + Phase 2-6 预声明）。头文件级核对已于 2026-08-06 完成，
# 本脚本补齐「.so 与头文件一致」的运行期验证（docs/v0.3-alpha-plan-zh.md P1.1）。
#
# 用法:
#   ./tools/check_musax_symbols.sh [SDK路径]      # 默认 $MUSA_INSTALL_PATH 或 /usr/local/musa
#
# 退出码: 0 = 全部 PASS; 1 = 存在 MISS

set -u

SDK="${1:-${MUSA_INSTALL_PATH:-${MUSA_HOME:-/usr/local/musa}}}"
LIB_DIR="$SDK/lib"

if [ ! -d "$LIB_DIR" ]; then
    echo "ERROR: SDK lib 目录不存在: $LIB_DIR" >&2
    exit 2
fi

# ── 待核对符号清单（库名:符号名）──────────────────────────────
SYMS="
# --- muBLAS（libmublas.so）: Phase 1 生命周期 + Phase 2 matmul/dot ---
libmublas.so:mublasCreate
libmublas.so:mublasDestroy
libmublas.so:mublasSetStream
libmublas.so:mublasGetVersion
libmublas.so:mublasSetPointerMode
libmublas.so:mublasSgemm
libmublas.so:mublasDgemm
libmublas.so:mublasCgemm
libmublas.so:mublasZgemm
libmublas.so:mublasSdot
libmublas.so:mublasDdot
libmublas.so:mublasCdotu
libmublas.so:mublasZdotu
libmublas.so:mublasSgemmStridedBatched
libmublas.so:mublasDgemmStridedBatched
# --- muSOLVER（libmusolver.so）: 无独立句柄,复用 mublasHandle_t; Phase 2/3 ---
libmusolver.so:musolverSgetrf
libmusolver.so:musolverDgetrf
libmusolver.so:musolverCgetrf
libmusolver.so:musolverZgetrf
libmusolver.so:musolverSgetrs
libmusolver.so:musolverDgetrs
libmusolver.so:musolverCgetrs
libmusolver.so:musolverZgetrs
libmusolver.so:musolverSgeqrf
libmusolver.so:musolverDgeqrf
libmusolver.so:musolverCgeqrf
libmusolver.so:musolverZgeqrf
libmusolver.so:musolverSorgqr
libmusolver.so:musolverDorgqr
libmusolver.so:musolverCungqr
libmusolver.so:musolverZungqr
libmusolver.so:musolverSgesvd
libmusolver.so:musolverDgesvd
libmusolver.so:musolverCgesvd
libmusolver.so:musolverZgesvd
libmusolver.so:musolverSsyevd
libmusolver.so:musolverDsyevd
libmusolver.so:musolverSgetrf_bufferSize
libmusolver.so:musolverDgetrf_bufferSize
libmusolver.so:musolverCgetrf_bufferSize
libmusolver.so:musolverZgetrf_bufferSize
libmusolver.so:musolverSgetrs_bufferSize
libmusolver.so:musolverDgetrs_bufferSize
libmusolver.so:musolverCgetrs_bufferSize
libmusolver.so:musolverZgetrs_bufferSize
libmusolver.so:musolverSgeqrf_bufferSize
libmusolver.so:musolverDgeqrf_bufferSize
libmusolver.so:musolverSgesvd_bufferSize
libmusolver.so:musolverDgesvd_bufferSize
libmusolver.so:musolverSsyevd_bufferSize
libmusolver.so:musolverDsyevd_bufferSize
# --- muRAND（libmurand.so）: Phase 4 random 套件 ---
libmurand.so:murandCreateGenerator
libmurand.so:murandDestroyGenerator
libmurand.so:murandSetStream
libmurand.so:murandGetVersion
libmurand.so:murandSetPseudoRandomGeneratorSeed
libmurand.so:murandSetGeneratorOffset
libmurand.so:murandGenerateUniform
libmurand.so:murandGenerateUniformDouble
libmurand.so:murandGenerateNormal
libmurand.so:murandGenerateNormalDouble
# --- muFFT（libmufft.so）: Phase 5 fft 套件 ---
libmufft.so:mufftCreate
libmufft.so:mufftDestroy
libmufft.so:mufftSetStream
libmufft.so:mufftGetVersion
libmufft.so:mufftPlan1d
libmufft.so:mufftPlan2d
libmufft.so:mufftPlan3d
libmufft.so:mufftPlanMany
libmufft.so:mufftExecC2C
libmufft.so:mufftExecR2C
libmufft.so:mufftExecC2R
libmufft.so:mufftExecZ2Z
libmufft.so:mufftExecD2Z
libmufft.so:mufftExecZ2D
# --- muSPARSE（libmusparse.so）: Phase 6 sparse 套件 ---
libmusparse.so:musparseCreate
libmusparse.so:musparseDestroy
libmusparse.so:musparseSetStream
libmusparse.so:musparseGetVersion
libmusparse.so:musparseCreateCsr
libmusparse.so:musparseCreateCoo
libmusparse.so:musparseCreateDnVec
libmusparse.so:musparseCreateDnMat
libmusparse.so:musparseDestroySpMat
libmusparse.so:musparseDestroyDnVec
libmusparse.so:musparseDestroyDnMat
libmusparse.so:musparseSpMV
libmusparse.so:musparseSpMM
"

declare -A NM_CACHE   # 库名 -> 符号表临时文件
pass=0
miss=0
total=0
missing_list=""

lookup() {
    local lib="$1"
    if [ -z "${NM_CACHE[$lib]:-}" ]; then
        local so="$LIB_DIR/$lib"
        if [ ! -e "$so" ]; then
            NM_CACHE[$lib]="__MISSING_LIB__"
        else
            local tmp
            tmp="$(mktemp)"
            nm -D --defined-only "$so" 2>/dev/null | awk '{print $NF}' | sort -u > "$tmp"
            NM_CACHE[$lib]="$tmp"
        fi
    fi
    grep -qx "$2" "${NM_CACHE[$lib]}" 2>/dev/null
}

echo "MUSA-X 运行期符号审计  (SDK: $SDK)"
echo "=================================================="

current_lib=""
while IFS= read -r line; do
    # 跳过空行与注释
    case "$line" in
        ''|\#*) continue ;;
    esac
    lib="${line%%:*}"
    sym="${line#*:}"
    if [ "$lib" != "$current_lib" ]; then
        echo "--- $lib ---"
        current_lib="$lib"
    fi
    total=$((total + 1))
    if lookup "$lib" "$sym"; then
        printf '  PASS  %s\n' "$sym"
        pass=$((pass + 1))
    else
        printf '  MISS  %s\n' "$sym"
        miss=$((miss + 1))
        missing_list="$missing_list $lib:$sym"
    fi
done <<< "$SYMS"

# 清理临时文件
for f in "${NM_CACHE[@]}"; do
    [ "$f" != "__MISSING_LIB__" ] && rm -f "$f"
done

echo "=================================================="
echo "合计: $total  通过: $pass  缺失: $miss"
if [ "$miss" -gt 0 ]; then
    echo "缺失清单:$missing_list"
    echo "→ 按 v0.3 计划 P1.1,缺失例程需记入风险表并决策(替代/引入/推迟)"
    exit 1
fi
echo "全部 PASS:5 个 MUSA-X 库的运行期符号与头文件核对一致"
