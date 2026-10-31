"""Nexmon CSI payload parsing and Broadcom BCM4366c0 packed-float decoding (own implementation,
written from the public Nexmon CSI documentation: 12-bit mantissas, 6-bit signed exponent,
sign bits at bit 29 (real) and bit 17 (imag)).

Word layout (uint32, little endian):
  [31:30] unused | [29] sign(re) | [28:18] |re| mantissa (11b) | [17] sign(im)
  [16:6] |im| mantissa (11b) | [5:0] signed exponent
"""
import struct
import numpy as np

from .pcap_reader import read_pcap, udp_payload

NEXMON_MAGIC = 0x1111
HEADER_LEN = 18


def parse_nexmon_payload(payload):
    """Return ((rssi,fctl,mac,seq,core_stream,chanspec,chip), words) or None if not Nexmon CSI."""
    if payload is None or len(payload) < HEADER_LEN + 4:
        return None
    if struct.unpack("<H", payload[:2])[0] != NEXMON_MAGIC:
        return None
    rssi = struct.unpack("<b", payload[2:3])[0]
    fctl = payload[3]
    mac = bytes(payload[4:10])
    seq, css, chanspec, chip = struct.unpack("<HHHH", payload[10:18])
    n = (len(payload) - HEADER_LEN) // 4
    words = np.frombuffer(payload, dtype="<u4", count=n, offset=HEADER_LEN)
    return (rssi, fctl, mac, seq, css, chanspec, chip), words


def unpack_bcm4366c0(words, nman=12, nexp=6, nbits=10, autoscale=True):
    """words: (N, S) uint32 -> complex128 CSI (N, S). All-zero words -> 0.

    autoscale=True gives each packet a common exponent so its largest component has
    ~`nbits` bits (the chip's own AGC-style convention); this only applies a per-packet
    constant, so relative subcarrier amplitudes are unchanged.
    """
    w = np.asarray(words, dtype=np.uint32)
    iq_mask = (1 << (nman - 1)) - 1
    e_p = 1 << (nexp - 1)
    vi = ((w >> (nexp + nman)) & iq_mask).astype(np.int64)
    vq = ((w >> nexp) & iq_mask).astype(np.int64)
    e = (w & ((1 << nexp) - 1)).astype(np.int64)
    e = np.where(e >= e_p, e - 2 * e_p, e)
    sr = np.where((w >> (nexp + 2 * nman - 1)) & 1, -1.0, 1.0)
    si = np.where((w >> (nexp + nman - 1)) & 1, -1.0, 1.0)
    x = vi | vq
    nz = x > 0
    if autoscale:
        msb = np.where(nz, np.floor(np.log2(np.maximum(x, 1))).astype(np.int64) + 1, -10**6)
        maxbit = np.max(np.where(nz, e + msb, -10**6), axis=1, keepdims=True)
        shft = np.where(maxbit > -10**5, nbits - maxbit, 0)
    else:
        shft = 0
    ee = e + shft
    keep = nz & (ee >= -nman)
    scale = np.ldexp(1.0, ee.astype(np.int32))
    re = np.where(keep, sr * vi * scale, 0.0)
    im = np.where(keep, si * vq * scale, 0.0)
    return re + 1j * im


def decode_pcap(path, decode_cfg):
    """Decode one Room-A capture. Returns (csi complex (N,S) in temporal order, info dict)."""
    frames, linktype, trunc = read_pcap(path)
    words, seqs = [], []
    n_non_udp = n_non_nexmon = n_badlen = n_narrow_dropped = 0
    S = decode_cfg["n_subcarriers"]
    for _, fr in frames:
        pl = udp_payload(fr, linktype)
        if pl is None:
            n_non_udp += 1
            continue
        p = parse_nexmon_payload(pl)
        if p is None:
            n_non_nexmon += 1
            continue
        hdr, w = p
        if len(w) != S:
            n_badlen += 1
            continue
        if decode_cfg.get("stream_policy", "all") == "full_band_only" and hdr[3] == 0xFFFF:
            n_narrow_dropped += 1
            continue
        words.append(w); seqs.append(hdr[3])
    info = dict(n_frames=len(frames), n_truncated_records=trunc, n_non_udp=n_non_udp,
                n_non_nexmon=n_non_nexmon, n_wrong_subcarrier_count=n_badlen, n_narrow_dropped=n_narrow_dropped,
                n_seq_ffff=int(np.sum(np.array(seqs) == 0xFFFF)) if seqs else 0)
    if not words:
        return np.zeros((0, S), complex), info
    csi = unpack_bcm4366c0(np.stack(words), decode_cfg["nman"], decode_cfg["nexp"],
                           decode_cfg["nbits"], decode_cfg["autoscale"])
    return csi, info
