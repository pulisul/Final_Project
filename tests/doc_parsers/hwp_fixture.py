"""HWP(OLE2) 테스트용 최소 작성기 — IntellectAgent_AI/src/test_parsers_sim.py 에서 가져온 것(실제 한컴 한글 파일이 아니라 직접 만든 OLE)."""
import io
import struct
import zlib

from intellectagent import doc_parsers as dp


ENDOFCHAIN, FATSECT, FREESECT, NOSTREAM = 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFF, 0xFFFFFFFF


# ---------------------------------------------------------------------------
# 최소 OLE2(v3) 작성기: 스트림은 4096 바이트 이상으로 패딩해 mini-stream 을 피한다
# ---------------------------------------------------------------------------
def build_ole(streams: dict[str, bytes]) -> bytes:
    """streams: {"FileHeader": ..., "BodyText/Section0": ...} (2단계 경로까지)."""
    storages = sorted({k.split("/")[0] for k in streams if "/" in k})
    entries = [("Root Entry", 5, None)]            # (이름, 타입, 데이터)
    top = [(k, 2, streams[k]) for k in streams if "/" not in k]
    for s in storages:
        top.append((s, 1, None))
    # 디렉터리: 루트.child = 첫 항목, 이후 right 로 이어 붙인다 (정렬 순서는 olefile 순회에 영향 없음)
    flat = [entries[0]] + top
    child_of: dict[str, list[tuple[str, int, bytes | None]]] = {}
    for k, v in streams.items():
        if "/" in k:
            st, nm = k.split("/")
            child_of.setdefault(st, []).append((nm, 2, v))
    order = list(flat)
    for s in storages:
        order.extend(child_of[s])
    n_dir = len(order)
    dir_sectors = (n_dir + 3) // 4

    def pad(b: bytes) -> bytes:
        b = b + b"\0" * (-len(b) % 512)
        return b if len(b) >= 4096 else b + b"\0" * (4096 - len(b))

    data_blobs: dict[int, bytes] = {}
    for idx, (_n, t, d) in enumerate(order):
        if t == 2:
            data_blobs[idx] = d
    fat_sector_count = 1
    next_sector = 1 + dir_sectors
    starts: dict[int, int] = {}
    sizes: dict[int, int] = {}
    for idx, d in data_blobs.items():
        padded = pad(d)
        starts[idx], sizes[idx] = next_sector, len(padded)
        next_sector += len(padded) // 512
    total_sectors = next_sector
    assert total_sectors <= 128, "테스트용 작성기: FAT 1섹터(128개)까지만 지원"
    fat = [FREESECT] * 128
    fat[0] = FATSECT
    for i in range(dir_sectors):
        fat[1 + i] = (2 + i) if i < dir_sectors - 1 else ENDOFCHAIN
    for idx in data_blobs:
        s, n = starts[idx], sizes[idx] // 512
        for j in range(n):
            fat[s + j] = (s + j + 1) if j < n - 1 else ENDOFCHAIN

    # 디렉터리 트리 연결
    idx_of = {id(e): i for i, e in enumerate(order)}
    left = [NOSTREAM] * n_dir
    right = [NOSTREAM] * n_dir
    child = [NOSTREAM] * n_dir
    top_idx = [idx_of[id(e)] for e in top]
    if top_idx:
        child[0] = top_idx[0]
        for a, b in zip(top_idx, top_idx[1:]):
            right[a] = b
    for s in storages:
        sidx = next(i for i, e in enumerate(order) if e[0] == s and e[1] == 1)
        kids = [i for i, e in enumerate(order) if e in child_of[s]]
        child[sidx] = kids[0]
        for a, b in zip(kids, kids[1:]):
            right[a] = b

    dirbuf = b""
    for i, (name, t, _d) in enumerate(order):
        nm = name.encode("utf-16le") + b"\0\0"
        ent = nm.ljust(64, b"\0") + struct.pack("<H", len(nm)) + bytes([t, 1])
        ent += struct.pack("<III", left[i], right[i], child[i]) + b"\0" * 16 + struct.pack("<I", 0) + b"\0" * 16
        if t == 5:
            ent += struct.pack("<IQ", ENDOFCHAIN, 0)
        else:
            ent += struct.pack("<IQ", starts.get(i, 0), sizes.get(i, 0) if t == 2 else 0) if t == 2 else struct.pack("<IQ", 0, 0)
        assert len(ent) == 128, len(ent)
        dirbuf += ent
    dirbuf = dirbuf.ljust(dir_sectors * 512, b"\0")
    # 사용하지 않는 디렉터리 슬롯은 type 0 (빈 항목) 이면 되고, 자식/형제 포인터는 NOSTREAM 이어야 한다
    empty = b"\0" * 64 + struct.pack("<H", 0) + bytes([0, 0]) + struct.pack("<III", NOSTREAM, NOSTREAM, NOSTREAM) + b"\0" * 48
    assert len(empty) == 128
    used = n_dir * 128
    dirbuf = dirbuf[:used] + empty * ((dir_sectors * 512 - used) // 128)

    header = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\0" * 16
    header += struct.pack("<HHHHH", 0x3E, 3, 0xFFFE, 9, 6) + b"\0" * 6
    header += struct.pack("<IIIIIIIII", 0, fat_sector_count, 1, 0, 4096, ENDOFCHAIN, 0, ENDOFCHAIN, 0)
    difat = [0] + [FREESECT] * 108
    header += struct.pack("<109I", *difat)
    assert len(header) == 512
    body = struct.pack("<128I", *fat) + dirbuf
    for idx in sorted(data_blobs):
        body += pad(data_blobs[idx])
    return header + body


def hwp_rec(tag: int, payload: bytes, level: int = 0) -> bytes:
    size = len(payload)
    if size >= 0xFFF:
        return struct.pack("<I", tag | (level << 10) | (0xFFF << 20)) + struct.pack("<I", size) + payload
    return struct.pack("<I", tag | (level << 10) | (size << 20)) + payload


def hwp_para(text: str, extended_ctrl: bool = False) -> bytes:
    raw = b""
    if extended_ctrl:   # 개체 삽입(코드 11) = 총 16바이트: 뒤따르는 부가정보는 본문이 아니다
        raw += struct.pack("<H", 11) + b"\x41\x00" * 7
    raw += text.encode("utf-16le") + struct.pack("<H", 13)
    return hwp_rec(dp.HWPTAG_PARA_TEXT, raw)


def hwp_file_header(compressed=True, encrypted=False, distribution=False) -> bytes:
    flags = (1 if compressed else 0) | (2 if encrypted else 0) | (4 if distribution else 0)
    h = b"HWP Document File".ljust(32, b"\0") + struct.pack("<II", 0x05010000, flags)
    return h.ljust(256, b"\0")


def make_hwp(paras_per_section: list[list[str]], compressed=True, **flags) -> bytes:
    streams = {"FileHeader": hwp_file_header(compressed, **flags)}
    for i, paras in enumerate(paras_per_section):
        body = b"".join(hwp_rec(66, b"\0" * 4) + hwp_para(p, extended_ctrl=(j == 0)) for j, p in enumerate(paras))
        if compressed:
            c = zlib.compressobj(wbits=-15)
            body = c.compress(body) + c.flush()
        streams[f"BodyText/Section{i}"] = body
    return build_ole(streams)

