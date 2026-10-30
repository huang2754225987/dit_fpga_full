"""Load the one-block DDR image and start the pure-PL ZCU106 design."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", help="USB-UART COM port, e.g. COM5")
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("build/supra2_block13_ddr_traffic/block13_payload.bin"),
    )
    parser.add_argument("--base", type=lambda s: int(s, 0), default=0x00100000)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--completion-timeout", type=float, default=1800.0)
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        print("pyserial is required: python -m pip install pyserial", file=sys.stderr)
        return 2

    image = args.image.read_bytes()
    if not image or len(image) % 16:
        raise SystemExit(f"image size must be a non-zero multiple of 16 bytes: {len(image)}")
    beats = len(image) // 16

    with serial.Serial(args.port, args.baud, timeout=2, write_timeout=10) as port:
        time.sleep(0.2)
        port.reset_input_buffer()
        port.write(b"\x57" + args.base.to_bytes(4, "little") + beats.to_bytes(4, "little"))
        port.flush()

        for index in range(beats):
            payload = image[index * 16 : (index + 1) * 16]
            port.write(payload)
            port.flush()
            response = port.read(1)
            if response != b"\x06":
                raise RuntimeError(
                    f"DDR write failed at beat {index}/{beats}; FPGA response={response.hex() or 'timeout'}"
                )
            if (index + 1) % 16384 == 0 or index + 1 == beats:
                print(f"DDR load: {index + 1}/{beats} beats ({100*(index+1)/beats:.1f}%)")

        port.write(b"\x53")
        port.flush()
        if port.read(1) != b"\x06":
            raise RuntimeError("FPGA did not acknowledge the block start command")
        print("Block started; waiting for completion (FPGA returns 0x04 when done)...")
        port.timeout = args.completion_timeout
        response = port.read(1)
        if response != b"\x04":
            raise RuntimeError(f"block did not complete; FPGA response={response.hex() or 'timeout'}")
        print("SUPRA2_BLOCK_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
