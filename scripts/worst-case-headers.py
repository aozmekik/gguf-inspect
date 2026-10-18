"""Writes two adversarial but valid GGUF files for worst-case memory checks:

- many-kv.gguf: 2^20 metadata entries with one-byte values (22,020,120 bytes)
- many-tensors.gguf: 2^20 scalar f32 tensor infos, all at offset 0
  (33,554,561 bytes)

Headers made of many tiny records cost the most memory per byte of file; see
docs/benchmarks.md. Standard library only.

usage: python3 -I scripts/worst-case-headers.py OUTPUT_DIR
"""

import struct
import sys
from pathlib import Path

N = 1 << 20


def string(b: bytes) -> bytes:
    return struct.pack("<Q", len(b)) + b


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "many-kv.gguf", "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, N))
        for i in range(N):
            f.write(string(b"k%07d" % i) + struct.pack("<IB", 0, 1))  # u8 = 1
    with open(out / "many-tensors.gguf", "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, N, 1))
        f.write(string(b"general.architecture") + struct.pack("<I", 8) + string(b"x"))
        for i in range(N):
            f.write(string(b"t%07d" % i) + struct.pack("<IIQ", 0, 0, 0))  # 0 dims, f32, offset 0
        f.write(b"\0" * 64)


if __name__ == "__main__":
    main()
