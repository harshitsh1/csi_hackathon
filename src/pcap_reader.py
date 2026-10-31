"""Minimal classic-PCAP reader (struct only) + Ethernet/IPv4/UDP payload extraction.
The challenge allows general-purpose PCAP parsing; the CSI decoding is in csi_decode.py."""
import struct

_MAGIC = {0xA1B2C3D4: ("<", 1e-6), 0xD4C3B2A1: (">", 1e-6),
          0xA1B23C4D: ("<", 1e-9), 0x4D3CB2A1: (">", 1e-9)}


def read_pcap(path):
    """Return (list of (timestamp, frame_bytes), linktype, n_truncated_records)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 24:
        raise ValueError(f"{path}: too short for a pcap header")
    magic = struct.unpack("<I", data[:4])[0]
    if magic not in _MAGIC:
        raise ValueError(f"{path}: not a classic pcap file (magic {magic:#x}); pcapng unsupported")
    end, res = _MAGIC[magic]
    linktype = struct.unpack(end + "I", data[20:24])[0]
    off, pk, trunc = 24, [], 0
    while off + 16 <= len(data):
        sec, frac, caplen, _ = struct.unpack(end + "IIII", data[off:off + 16])
        off += 16
        if off + caplen > len(data):
            trunc += 1
            break
        pk.append((sec + frac * res, data[off:off + caplen]))
        off += caplen
    return pk, linktype, trunc


def udp_payload(frame, linktype=1):
    """UDP payload of an Ethernet/IPv4/UDP frame, else None."""
    if linktype != 1 or len(frame) < 42:
        return None
    et, o = struct.unpack(">H", frame[12:14])[0], 14
    if et == 0x8100:                       # 802.1Q tag
        et, o = struct.unpack(">H", frame[16:18])[0], 18
    if et != 0x0800:
        return None
    ihl = (frame[o] & 0x0F) * 4
    if frame[o + 9] != 17:
        return None
    u = o + ihl
    ulen = struct.unpack(">H", frame[u + 4:u + 6])[0]
    return frame[u + 8:u + ulen] if ulen >= 8 else None
