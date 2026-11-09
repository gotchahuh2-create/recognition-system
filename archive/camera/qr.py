"""Minimal QR code encoder (byte mode, error correction M, versions 1-15).

Only what the pairing page needs: encode a URL of up to ~400 bytes and draw
it as SVG. Written here so the camera server needs no extra package; checked
against OpenCV's QR decoder in ``tests/test_camera.py``.
"""

from __future__ import annotations

# (ec codewords per block, [(blocks, data codewords per block), ...]) for level M
_EC_M = {
    1: (10, [(1, 16)]), 2: (16, [(1, 28)]), 3: (26, [(1, 44)]), 4: (18, [(2, 32)]),
    5: (24, [(2, 43)]), 6: (16, [(4, 27)]), 7: (18, [(4, 31)]),
    8: (22, [(2, 38), (2, 39)]), 9: (22, [(3, 36), (2, 37)]),
    10: (26, [(4, 43), (1, 44)]), 11: (30, [(1, 50), (4, 51)]),
    12: (22, [(6, 36), (2, 37)]), 13: (22, [(8, 37), (1, 38)]),
    14: (24, [(4, 40), (5, 41)]), 15: (24, [(5, 41), (5, 42)]),
}
_ALIGN = {1: [], 2: [6, 18], 3: [6, 22], 4: [6, 26], 5: [6, 30], 6: [6, 34],
          7: [6, 22, 38], 8: [6, 24, 42], 9: [6, 26, 46], 10: [6, 28, 50],
          11: [6, 30, 54], 12: [6, 32, 58], 13: [6, 34, 62], 14: [6, 26, 46, 66],
          15: [6, 26, 48, 70]}

# GF(256) with polynomial 0x11d
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _gmul(a: int, b: int) -> int:
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


def _rs_ec(data: list[int], n: int) -> list[int]:
    gen = [1]
    for i in range(n):
        gen = [a ^ b for a, b in zip(gen + [0], [0] + [_gmul(g, _EXP[i]) for g in gen])]
    rem = list(data) + [0] * n
    for i in range(len(data)):
        c = rem[i]
        if c:
            for j in range(1, len(gen)):
                rem[i + j] ^= _gmul(gen[j], c)
    return rem[len(data):]


def _bch(value: int, poly: int, bits: int) -> int:
    v = value << bits
    top = poly.bit_length() - 1
    while v.bit_length() - 1 >= top:
        v ^= poly << (v.bit_length() - 1 - top)
    return (value << bits) | v


def encode(text: str) -> list[list[bool]]:
    """QR matrix (True = dark module) for ``text`` (UTF-8, byte mode, level M)."""
    data = text.encode("utf-8")
    for ver in range(1, 16):
        ec, groups = _EC_M[ver]
        cap = sum(b * d for b, d in groups)
        cnt_bits = 8 if ver < 10 else 16
        if 4 + cnt_bits + 8 * len(data) <= cap * 8:
            break
    else:
        raise ValueError(f"text too long for a QR code here ({len(data)} bytes)")
    bits: list[int] = []

    def put(v: int, n: int):
        bits.extend((v >> (n - 1 - i)) & 1 for i in range(n))
    put(0b0100, 4)
    put(len(data), cnt_bits)
    for b in data:
        put(b, 8)
    put(0, min(4, cap * 8 - len(bits)))
    while len(bits) % 8:
        bits.append(0)
    cw = [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)]
    pad = (0xEC, 0x11)
    i = 0
    while len(cw) < cap:
        cw.append(pad[i % 2])
        i += 1
    # blocks + interleave
    blocks, k = [], 0
    for nb, nd in groups:
        for _ in range(nb):
            blocks.append(cw[k:k + nd])
            k += nd
    ecs = [_rs_ec(b, ec) for b in blocks]
    final = []
    for i in range(max(len(b) for b in blocks)):
        final += [b[i] for b in blocks if i < len(b)]
    for i in range(ec):
        final += [e[i] for e in ecs]
    stream = [(c >> (7 - j)) & 1 for c in final for j in range(8)]
    stream += [0] * ([0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 3, 3][ver] if ver < 16 else 0)

    n = 17 + 4 * ver
    m = [[False] * n for _ in range(n)]
    fixed = [[False] * n for _ in range(n)]

    def setf(r, c, v):
        m[r][c] = v
        fixed[r][c] = True

    for r0, c0 in ((0, 0), (0, n - 7), (n - 7, 0)):
        for r in range(-1, 8):
            for c in range(-1, 8):
                rr, cc = r0 + r, c0 + c
                if 0 <= rr < n and 0 <= cc < n:
                    v = (0 <= r <= 6 and 0 <= c <= 6 and
                         (r in (0, 6) or c in (0, 6) or (2 <= r <= 4 and 2 <= c <= 4)))
                    setf(rr, cc, v)
    for i in range(8, n - 8):
        setf(6, i, i % 2 == 0)
        setf(i, 6, i % 2 == 0)
    al = _ALIGN[ver]
    for r in al:
        for c in al:
            if (r, c) in ((al[0], al[0]), (al[0], al[-1]), (al[-1], al[0])):
                continue                       # would overlap a finder pattern
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    setf(r + dr, c + dc, max(abs(dr), abs(dc)) != 1)
    setf(n - 8, 8, True)                       # dark module
    for i in range(9):                         # reserve format areas
        if not fixed[8][i]:
            fixed[8][i] = True
        if not fixed[i][8]:
            fixed[i][8] = True
    for i in range(8):
        fixed[8][n - 1 - i] = True
        fixed[n - 1 - i][8] = True
    if ver >= 7:
        vbits = _bch(ver, 0x1F25, 12)
        for i in range(18):
            v = bool((vbits >> i) & 1)
            setf(n - 11 + i % 3, i // 3, v)
            setf(i // 3, n - 11 + i % 3, v)

    # data placement (zig-zag)
    idx = 0
    col = n - 1
    up = True
    while col > 0:
        if col == 6:
            col -= 1
        rows = range(n - 1, -1, -1) if up else range(n)
        for r in rows:
            for c in (col, col - 1):
                if not fixed[r][c]:
                    m[r][c] = bool(stream[idx]) if idx < len(stream) else False
                    idx += 1
        up = not up
        col -= 2

    masks = [lambda r, c: (r + c) % 2 == 0, lambda r, c: r % 2 == 0, lambda r, c: c % 3 == 0,
             lambda r, c: (r + c) % 3 == 0, lambda r, c: (r // 2 + c // 3) % 2 == 0,
             lambda r, c: (r * c) % 2 + (r * c) % 3 == 0,
             lambda r, c: ((r * c) % 2 + (r * c) % 3) % 2 == 0,
             lambda r, c: ((r + c) % 2 + (r * c) % 3) % 2 == 0]

    def with_mask(k):
        mm = [row[:] for row in m]
        for r in range(n):
            for c in range(n):
                if not fixed[r][c] and masks[k](r, c):
                    mm[r][c] = not mm[r][c]
        fmt = _bch((0b00 << 3) | k, 0x537, 10) ^ 0x5412      # level M = 00
        fb = [(fmt >> i) & 1 for i in range(15)]
        for i in range(6):                     # first copy, around the top-left finder
            mm[i][8] = bool(fb[i])
        mm[7][8] = bool(fb[6])
        mm[8][8] = bool(fb[7])
        mm[8][7] = bool(fb[8])
        for i in range(9, 15):
            mm[8][14 - i] = bool(fb[i])
        for i in range(8):                     # second copy, split
            mm[8][n - 1 - i] = bool(fb[i])
        for i in range(8, 15):
            mm[n - 15 + i][8] = bool(fb[i])
        mm[n - 8][8] = True
        return mm

    best, best_pen = None, None
    for k in range(8):
        mm = with_mask(k)
        pen = _penalty(mm)
        if best_pen is None or pen < best_pen:
            best, best_pen = mm, pen
    return best


def _penalty(m) -> int:
    n = len(m)
    p = 0
    for grid in (m, [list(x) for x in zip(*m)]):
        for row in grid:
            run, prev = 0, None
            for v in row:
                if v == prev:
                    run += 1
                else:
                    if run >= 5:
                        p += run - 2
                    run, prev = 1, v
            if run >= 5:
                p += run - 2
            s = "".join("1" if v else "0" for v in row)
            p += 40 * (s.count("10111010000") + s.count("00001011101"))
    for r in range(n - 1):
        for c in range(n - 1):
            if m[r][c] == m[r][c + 1] == m[r + 1][c] == m[r + 1][c + 1]:
                p += 3
    dark = sum(v for row in m for v in row)
    p += 10 * (abs(dark * 20 - n * n * 10) // (n * n))
    return p


def svg(text: str, scale: int = 6, border: int = 4, dark: str = "#000", light: str = "#fff") -> str:
    m = encode(text)
    n = len(m)
    size = (n + 2 * border) * scale
    path = "".join(f"M{(c + border) * scale},{(r + border) * scale}h{scale}v{scale}h-{scale}z"
                   for r in range(n) for c in range(n) if m[r][c])
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
            f'width="{size}" height="{size}" shape-rendering="crispEdges">'
            f'<rect width="100%" height="100%" fill="{light}"/><path d="{path}" fill="{dark}"/></svg>')
