"""Read compact radar windows with bounded memory and explicit short-read checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class PackedRadar:
    """Read contiguous half-precision windows without keeping a file mapping open."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)
        with open(self.path, "rb") as stream:
            major, _ = np.lib.format.read_magic(stream)
            if major == 1:
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
            else:
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
            self.offset = stream.tell()
        if fortran or dtype != np.float16:
            raise ValueError(f"{self.path}: radar storage must be C-contiguous float16")
        self.shape = tuple(shape)
        self.frame_items = int(np.prod(shape[1:]))

    def __getitem__(self, selection: slice) -> np.ndarray:
        start, stop, step = selection.indices(self.shape[0])
        if step != 1:
            raise ValueError("Radar windows require a contiguous slice")
        count = max(0, stop - start)
        output = np.empty((count,) + self.shape[1:], np.float16)
        if count == 0:
            return output
        with open(self.path, "rb") as stream:
            stream.seek(self.offset + start * self.frame_items * 2)
            received = stream.readinto(memoryview(output).cast("B"))
        if received != output.nbytes:
            raise ValueError(
                f"{self.path}: truncated radar window; expected {output.nbytes} bytes, "
                f"received {received}"
            )
        return output
