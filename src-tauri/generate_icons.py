"""Generate minimal Meruem icon assets (PNG + ICO) without external deps."""
import struct
import zlib
from pathlib import Path

SIZE = 256
OUT = Path(__file__).parent

BG = (11, 15, 20, 255)
GOLD = (234, 179, 8, 255)


def chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def make_png(path: Path) -> None:
    rows = bytearray()
    for y in range(SIZE):
        rows.append(0)  # filter: none
        for x in range(SIZE):
            # gold diamond "M" motif
            color = BG
            dx = abs(x - SIZE // 2)
            dy = abs(y - SIZE // 2)
            if dx + dy < SIZE // 3:
                color = GOLD
            rows.extend(color)
    raw = bytes(rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def make_ico(png_path: Path, ico_path: Path) -> None:
    png = png_path.read_bytes()
    header = struct.pack("<HHH", 0, 1, 1)
    entry = struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(png), 22)
    ico_path.write_bytes(header + entry + png)


def main() -> None:
    (OUT / "icons").mkdir(parents=True, exist_ok=True)
    make_png(OUT / "icons" / "icon.png")
    make_ico(OUT / "icons" / "icon.png", OUT / "icons" / "icon.ico")
    print("icons generated:", OUT / "icons")


if __name__ == "__main__":
    main()
