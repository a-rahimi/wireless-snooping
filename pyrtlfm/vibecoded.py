import sys
import numpy as np
import pickle
from typing import NamedTuple, List, Tuple
import types
from scipy.signal import butter, filtfilt
from scipy.stats import linregress
import matplotlib.pyplot as plt


# Define DemodPacket matching the one in pipeline.py
class DemodPacket(NamedTuple):
    timestamp: float
    iq_gated: np.ndarray


def phase_velocity(packet: DemodPacket) -> np.ndarray:
    return np.angle(packet.iq_gated[1:] * np.conj(packet.iq_gated[:-1])) / np.pi


def amplitude(packet: DemodPacket) -> np.ndarray:
    return np.abs(packet.iq_gated[:-1])


def setup_pickle_mock():
    # Mock the module structure so pickle can find DemodPacket
    # The pickle likely refers to 'pyrtlfm.pipeline.DemodPacket' OR 'pipeline.DemodPacket'
    # We will try to catch both.
    fake_pipeline = types.ModuleType("pyrtlfm.pipeline")
    fake_pipeline.DemodPacket = DemodPacket

    # Ensure pyrtlfm package exists in sys.modules
    fake_pyrtlfm = types.ModuleType("pyrtlfm")
    fake_pyrtlfm.pipeline = fake_pipeline
    sys.modules["pyrtlfm"] = fake_pyrtlfm
    sys.modules["pyrtlfm.pipeline"] = fake_pipeline

    # Also mock 'pipeline' top level just in case
    sys.modules["pipeline"] = fake_pipeline


def is_contiguous_packet(packet: DemodPacket) -> bool:
    """
    Determine if the packet has one contiguous high-amplitude region.
    """
    amp = amplitude(packet)
    # Handle empty packet
    if len(amp) == 0:
        return False

    threshold = 0.5 * np.max(amp)

    # Get indices where amplitude is above threshold
    high_indices = np.nonzero(amp > threshold)[0]

    if len(high_indices) == 0:
        return False

    # Trim edges as done in pipeline.py
    if len(high_indices) > 20:
        high_indices = high_indices[10:-10]

    if len(high_indices) == 0:
        return False

    # Check for continuity
    if np.any(np.diff(high_indices) > 1):
        return False

    return True


def fsk_filtered(
    packet: DemodPacket,
    cutoff_hz: float = 50_000,
    sample_rate: float = 2.4e6,
    order: int = 5,
) -> np.ndarray:
    """FSK decode with low-pass filtering."""
    # Note: we use phase_velocity function defined above which takes DemodPacket
    # but we can also just pass the pv array if we adapt it.
    # But for consistency with pipeline.py signature:
    pv = phase_velocity(packet)
    return filtfilt(
        *butter(order, cutoff_hz / (sample_rate / 2), btype="low"),
        x=pv,
    )


def decode_bits(pv: np.ndarray):
    # Center
    pv = pv - np.median(pv)

    # Find crossings in preamble (first ~500 samples)
    preamble_limit = min(1000, len(pv))

    # Get crossings
    crossings = np.where(np.diff(np.sign(pv[:preamble_limit])))[0]

    # Filter crossings to remove noise
    clean_crossings = []
    if len(crossings) > 0:
        clean_crossings.append(crossings[0])
        for x in crossings[1:]:
            if x - clean_crossings[-1] > 10:  # Min distance ~ half a bit width
                clean_crossings.append(x)

    if len(clean_crossings) < 10:
        return None, None  # Not enough preamble

    # Fit line to find bit width and offset
    res = linregress(np.arange(len(clean_crossings)), clean_crossings)
    bit_width = res.slope
    offset = res.intercept

    # print(f"  Bit width: {bit_width:.2f}, Offset: {offset:.2f}")

    total_samples = len(pv)

    # Safety check on bit_width
    if bit_width < 5:
        return None, None

    num_bits = int((total_samples - offset) / bit_width)

    bits = []

    for i in range(num_bits):
        # Determine bit window
        start = int(offset + i * bit_width)
        end = int(offset + (i + 1) * bit_width)

        if start < 0:
            start = 0
        if end > total_samples:
            break
        if start >= end:
            continue

        # Average PV
        bit_val = np.mean(pv[start:end])
        # If the preamble is 101010, the PV should toggle positive/negative.
        # bit 1 if val > 0, else 0.
        # But polarity might be inverted. Preamble 1010 or 0101?
        # Standard FSK: high freq -> 1, low freq -> 0 usually.
        bit = 1 if bit_val > 0 else 0
        bits.append(bit)

    return bits, bit_width


def find_payload_start(bits: List[int]) -> int:
    # Preamble is 101010... or 010101...
    # Find first violation (00 or 11) after some initial stability

    consecutive_alternating = 0
    preamble_found = False

    for i in range(1, len(bits)):
        if bits[i] != bits[i - 1]:
            consecutive_alternating += 1
            if consecutive_alternating > 8:
                preamble_found = True
        else:
            # Violation found (00 or 11)
            if preamble_found:
                # We found the sync word / payload start.
                # Usually the violation IS the sync word or part of it.
                return i
            consecutive_alternating = 0

    return -1


def analyze_packet(packet: DemodPacket, packet_id: int):
    if not is_contiguous_packet(packet):
        return None, None

    # Get filtered FSK for the high amplitude region
    fsk = fsk_filtered(packet)

    amp = amplitude(packet)
    threshold = 0.5 * np.max(amp)
    high_indices = np.nonzero(amp > threshold)[0]

    if len(high_indices) <= 20:
        return None, None

    packet_samples = high_indices[-1] - high_indices[0] + 1
    bits, width = decode_bits(fsk[high_indices[10] : high_indices[-10]])

    if not bits:
        print(
            f"Packet {packet_id} (t={packet.timestamp:.3f}s): {packet_samples} samples — Failed to decode bits (no preamble found?)"
        )
        return None, None

    print(
        f"Packet {packet_id} (t={packet.timestamp:.3f}s): {packet_samples} samples, {width:.1f} samples/bit — ",
        end="",
    )
    payload_start = find_payload_start(bits)
    if payload_start == -1:
        print("No preamble/payload boundary found.")
        return None, None

    payload_bits = bits[payload_start:]

    # Format payload bits
    payload_str = "".join(map(str, payload_bits))
    # Try converting to hex if length is multiple of 8
    # Align to 8 bits
    num_bytes = len(payload_bits) // 8
    tp_info = None

    if num_bytes > 0:
        byte_vals = []
        for i in range(num_bytes):
            # Big endian or little endian?
            # Network order is usually Big Endian (MSB first)
            b_str = payload_str[i * 8 : (i + 1) * 8]
            byte_vals.append(int(b_str, 2))

        hex_str = " ".join(f"{b:02X}" for b in byte_vals)
        print(hex_str, end="")

        # Check for ThermoPro signature (18 BF ...)
        if len(byte_vals) >= 5 and byte_vals[0] == 0x18 and byte_vals[1] == 0xBF:
            # Protocol: 18 BF [ID] FF [TEMP] [CS] ...
            # Temp = Byte4 / 2.0 (Fahrenheit)

            id_byte = byte_vals[2]
            const_byte = byte_vals[3]  # Should be FF
            raw_temp = byte_vals[4]
            checksum_byte = byte_vals[5] if len(byte_vals) > 5 else None

            temp_f = raw_temp / 2.0

            # Store info for table
            tp_info = {
                "id": packet_id,
                "t": packet.timestamp,
                "byte2": id_byte,
                "byte3": const_byte,
                "raw_temp": raw_temp,
                "checksum": checksum_byte,
                "temp_f": temp_f,
                "bytes": hex_str,
            }

        print()

    return bits, tp_info


def main():
    setup_pickle_mock()
    with open("pyrtlfm/capture.pkl", "rb") as f:
        packets = pickle.load(f)

    # Normalize timestamps so the first packet starts at t=0
    start_time = packets[0].timestamp
    packets = [p._replace(timestamp=p.timestamp - start_time) for p in packets]

    print(f"Loaded {len(packets)} packets.")

    tp_packets = []

    for i, packet in enumerate(packets):
        _, tp_info = analyze_packet(packet, i)
        if tp_info:
            tp_packets.append(tp_info)

    print("\n" + "=" * 80)
    print("ThermoPro Packet Summary (18 BF ...)")
    print("=" * 80)
    print(
        f"{'Pkt':<4} | {'Time (s)':<9} | {'ID':<3} | {'Fix':<3} | {'RawT':<4} | {'CS':<3} | {'Temp F':<7} | {'Data'}"
    )
    print("-" * 80)

    for p in tp_packets:
        cs_str = f"{p['checksum']:02X}" if p["checksum"] is not None else "--"
        print(
            f"{p['id']:<4} | {p['t']:<9.3f} | {p['byte2']:02X}  | {p['byte3']:02X}  | {p['raw_temp']:02X}   | {cs_str:<3} | {p['temp_f']:<7.1f} | {p['bytes']}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
