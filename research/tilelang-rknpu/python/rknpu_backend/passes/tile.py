# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure geometry helpers for the fixed 256-wide MVP tiling recipe."""
from __future__ import annotations

from dataclasses import dataclass

from ..errors import RK_E_SHAPE, reject

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


def output_tiles(m: int, n: int) -> tuple[OutputTile, ...]:
    if not isinstance(m, int) or not isinstance(n, int) or not 1 <= m <= 4096 or \
            not 1 <= n <= 4096:
        reject(RK_E_SHAPE, "M/N must be in 1..4096", stage="tile",
               actual=[m, n])
    tiles: list[OutputTile] = []
    for mi, m0 in enumerate(range(0, m, TILE_LIMIT)):
        rows = min(TILE_LIMIT, m - m0)
        for ni, n0 in enumerate(range(0, n, TILE_LIMIT)):
            cols = min(TILE_LIMIT, n - n0)
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
