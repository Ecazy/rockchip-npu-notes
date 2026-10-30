# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure geometry helpers for the fixed 256-wide MVP tiling recipe."""
from __future__ import annotations

from dataclasses import dataclass

from ..errors import RK_E_NUMERIC_CONTRACT, RK_E_SHAPE, RK_E_UNSUPPORTED, \
    reject

TILE_LIMIT = 256
K_TILE = 256


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@dataclass(frozen=True)
class KSlice:
    index: int
    k0: int
    extent: int


@dataclass(frozen=True)
class MNTileSchedule:
    """The one supported scheduling contract for an RKIR matmul.

    K blocking and accumulation order are deliberately not user-controlled in
    this milestone: the planner always emits ascending 256-wide K slices with
    ordered FP16 writeback.
    """

    tile_m: int = TILE_LIMIT
    tile_n: int = TILE_LIMIT
    order: str = "m_major"

    def to_dict(self) -> dict:
        return {"kind": "mn_tiles_v1", "tile_m": self.tile_m,
                "tile_n": self.tile_n, "order": self.order,
                "k_tile": K_TILE,
                "accumulation": "ordered_ascending_f16_writeback"}


MN_TILE_ORDERS = {"m_major", "n_major"}
SCHEDULE_KEYS = {"kind", "tile_m", "tile_n", "order", "k_tile",
                 "accumulation"}


def mn_schedule_from_attrs(attrs: dict, *, where: str = "matmul") -> MNTileSchedule:
    """Decode the explicit ``source_schedule`` RKIR attribute.

    Absence means the historical 256x256, M-major planner default.  Once a
    schedule is present every key is validated; unknown keys and unsupported
    orders are rejected instead of being silently ignored.
    """
    raw = attrs.get("source_schedule")
    if raw is None:
        return MNTileSchedule()
    if not isinstance(raw, dict):
        reject(RK_E_UNSUPPORTED, f"{where}.source_schedule must be an object",
               stage="schedule", actual=type(raw).__name__)
    unknown = sorted(set(raw) - SCHEDULE_KEYS)
    missing = sorted({"kind", "tile_m", "tile_n", "order", "k_tile",
                      "accumulation"} - set(raw))
    if unknown or missing:
        reject(RK_E_UNSUPPORTED,
               f"{where}.source_schedule has unsupported or missing fields",
               stage="schedule", actual={"unknown": unknown,
                                         "missing": missing})
    if raw["kind"] != "mn_tiles_v1":
        reject(RK_E_UNSUPPORTED, f"unsupported {where} schedule kind",
               stage="schedule", actual=raw["kind"],
               required="mn_tiles_v1")
    for key in ("tile_m", "tile_n", "k_tile"):
        value = raw[key]
        if not isinstance(value, int) or isinstance(value, bool):
            reject(RK_E_UNSUPPORTED,
                   f"{where}.source_schedule.{key} must be an integer",
                   stage="schedule", actual=value)
    if raw["k_tile"] != K_TILE:
        reject(RK_E_UNSUPPORTED,
               f"K scheduling is fixed at {K_TILE} in this milestone",
               stage="schedule", actual=raw["k_tile"], required=K_TILE)
    if raw["accumulation"] != "ordered_ascending_f16_writeback":
        reject(RK_E_UNSUPPORTED,
               "unsupported K accumulation order; only ordered ascending "
               "FP16 writeback is admitted", stage="schedule",
               actual=raw["accumulation"])
    if not 1 <= raw["tile_m"] <= TILE_LIMIT or \
            not 1 <= raw["tile_n"] <= TILE_LIMIT:
        reject(RK_E_UNSUPPORTED,
               f"M/N tile sizes must be within 1..{TILE_LIMIT}",
               stage="schedule", actual={"tile_m": raw["tile_m"],
                                         "tile_n": raw["tile_n"]})
    if raw["order"] not in MN_TILE_ORDERS:
        reject(RK_E_UNSUPPORTED,
               "unsupported M/N traversal order", stage="schedule",
               actual=raw["order"], required=sorted(MN_TILE_ORDERS))
    # A scheduled source must state rounding that is equivalent to the target
    # per-K FP16 writeback.  A final-only FP32 narrowing is a different
    # numerical program and must not silently inherit this schedule.
    rounding = attrs.get("source_k_rounding")
    if rounding not in {None, "explicit_per_k_fp16", "single_k_tile_fp16"}:
        reject(RK_E_NUMERIC_CONTRACT,
               "scheduled source has a different K-rounding contract",
               stage="schedule", actual=rounding,
               required=["explicit_per_k_fp16", "single_k_tile_fp16"])
    # A source K loop, if present, must agree with the fixed lowering contract.
    k_loop = attrs.get("source_k_loop")
    if k_loop is not None:
        if not isinstance(k_loop, dict):
            reject(RK_E_UNSUPPORTED, "source_k_loop must be an object",
                   stage="schedule")
        block = k_loop.get("source_k_block")
        if block is not None and block != K_TILE:
            reject(RK_E_UNSUPPORTED,
                   f"source K block must be {K_TILE}; changing it would change "
                   "the reduction order", stage="schedule", actual=block,
                   required=K_TILE)
        lowering = k_loop.get("lowering_k_tile")
        if lowering is not None and lowering != K_TILE:
            reject(RK_E_UNSUPPORTED,
                   f"lowering K tile must be {K_TILE}", stage="schedule",
                   actual=lowering, required=K_TILE)
    return MNTileSchedule(raw["tile_m"], raw["tile_n"], raw["order"])


@dataclass(frozen=True)
class OutputTile:
    mi: int
    ni: int
    m0: int
    n0: int
    rows: int
    cols: int


def k_slices(k: int) -> tuple[KSlice, ...]:
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 4096:
        reject(RK_E_SHAPE, "K must be in 1..4096", stage="tile", actual=k)
    return tuple(KSlice(i, i * K_TILE, min(K_TILE, k - i * K_TILE))
                 for i in range((k + K_TILE - 1) // K_TILE))


def output_tiles(m: int, n: int, *,
                 tile_m: int = TILE_LIMIT, tile_n: int = TILE_LIMIT,
                 order: str = "m_major") -> tuple[OutputTile, ...]:
    if not isinstance(m, int) or not isinstance(n, int) or not 1 <= m <= 4096 or \
            not 1 <= n <= 4096:
        reject(RK_E_SHAPE, "M/N must be in 1..4096", stage="tile",
               actual=[m, n])
    if not isinstance(tile_m, int) or not isinstance(tile_n, int) or \
            1 > tile_m or 1 > tile_n or tile_m > TILE_LIMIT or \
            tile_n > TILE_LIMIT:
        reject(RK_E_UNSUPPORTED,
               f"M/N tile sizes must be within 1..{TILE_LIMIT}", stage="tile",
               actual=[tile_m, tile_n])
    if order not in MN_TILE_ORDERS:
        reject(RK_E_UNSUPPORTED, "unsupported M/N traversal order",
               stage="tile", actual=order, required=sorted(MN_TILE_ORDERS))
    m_tiles = []
    for mi, m0 in enumerate(range(0, m, tile_m)):
        m_tiles.append((mi, m0, min(tile_m, m - m0)))
    n_tiles = []
    for ni, n0 in enumerate(range(0, n, tile_n)):
        n_tiles.append((ni, n0, min(tile_n, n - n0)))
    tiles: list[OutputTile] = []
    if order == "m_major":
        pairs = ((m, n) for m in m_tiles for n in n_tiles)
    else:
        pairs = ((m, n) for n in n_tiles for m in m_tiles)
    for (mi, m0, rows), (ni, n0, cols) in pairs:
        tiles.append(OutputTile(mi, ni, m0, n0, rows, cols))
    return tuple(tiles)


def cube_layout(rows: int, cols: int) -> tuple[int, int, int]:
    """Mirror co_tensor_matrix: physical rows and columns, payload bytes."""
    if not 1 <= rows <= 256 or not 1 <= cols <= 256:
        reject(RK_E_SHAPE, "cube tile extents must be within 1..256", stage="tile",
               actual=[rows, cols])
    physical_rows = round_up(max(rows, 12), 4)
    physical_cols = round_up(cols, 32)
    return physical_rows, physical_cols, physical_rows * physical_cols * 2


def input_layout(rows: int, kt: int) -> tuple[int, int, int]:
    if not 1 <= rows <= 256 or not 1 <= kt <= K_TILE:
        reject(RK_E_SHAPE, "input tile extents must be within 1..256", stage="tile",
               actual=[rows, kt])
    physical_rows = round_up(max(rows, 12), 4)
    physical_cols = round_up(kt, 32)
    return physical_rows, physical_cols, physical_rows * physical_cols * 2


def weight_layout(cols: int, kt: int) -> tuple[int, int, int]:
    if not 1 <= cols <= 256 or not 1 <= kt <= K_TILE:
        reject(RK_E_SHAPE, "weight tile extents must be within 1..256", stage="tile",
               actual=[cols, kt])
    physical_rows = round_up(cols, 32)
    physical_cols = round_up(kt, 32)
    return physical_rows, physical_cols, physical_rows * physical_cols * 2


def lut_chunks(physical_elements: int) -> tuple[tuple[int, int], ...]:
    """Split a physical cube into legal linear LUT views.

    The existing emitter contract is ``physical_elements/8 < 8191``.  The
    established graph implementation uses 32768-element chunks, which also keeps
    each offset 16-byte aligned and each linear extent below ``co_tensor_linear``'s
    65520-element cap.
    """
    if physical_elements <= 0:
        reject(RK_E_SHAPE, "LUT physical element count must be positive", stage="tile")
    chunks: list[tuple[int, int]] = []
    offset = 0
    while offset < physical_elements:
        count = min(32768, physical_elements - offset)
        if count > 65520 or (count * 2) % 16:
            reject(RK_E_SHAPE, "LUT chunk exceeds the linear-view contract",
                   stage="tile")
        chunks.append((offset, count))
        offset += count
    return tuple(chunks)
