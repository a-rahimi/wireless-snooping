"""Live SDR capture loop: squelch-gated IQ packet collection."""

import threading
import time

import numpy as np
from rtlsdr import RtlSdr

import packets


def _ensure_complex_ndarray(samples) -> np.ndarray:
    """Convert pyrtlsdr callback samples to 1D complex ndarray."""
    arr = np.asarray(samples, dtype=np.complex64)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def capture_packets() -> list[packets.DemodPacket]:
    """Run the pipeline at 915 MHz, 2.4 Msps.

    Press Ctrl+C to stop.
    """
    packets_list: list[packets.DemodPacket] = []

    def packet_callback(iq, ctx) -> None:
        iq = _ensure_complex_ndarray(iq)
        if iq.size < 2:
            return

        if not packets.squelch(iq, 0.1):
            return

        timestamp = time.time()
        packets_list.append(packets.DemodPacket(timestamp, iq.copy(), sdr.sample_rate))
        print("Packet received", timestamp)

    def run(sdr: RtlSdr) -> None:
        try:
            sdr.read_samples_async(packet_callback, num_samples=16384)
        finally:
            print("Closing SDR...")
            sdr.cancel_read_async()
            sdr.close()

    sdr = RtlSdr(device_index=0)
    sdr.sample_rate = 300_000
    sdr.center_freq = 915e6
    sdr.gain = "auto"

    thread = threading.Thread(target=run, args=(sdr,), daemon=False)
    thread.start()
    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nStopping...")
        sdr.cancel_read_async()
        thread.join(timeout=5.0)
    print("Stopped.")
    return packets_list
