import argparse
import asyncio
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice
from bleak.exc import BleakError
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

try:
    from matplotlib import font_manager
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    font_manager = None
    MATPLOTLIB_AVAILABLE = False


DEFAULT_ADDRESS = "00:00:00:00:00:00"
CONTROL_WRITE_UUID = "0000ae01-0000-1000-8000-00805f9b34fb"
NOTIFY_UUID = "0000ae02-0000-1000-8000-00805f9b34fb"
DATA_WRITE_UUID = "0000ae03-0000-1000-8000-00805f9b34fb"

PRINTER_WIDTH_PIXELS = 384
PRINTER_WIDTH_BYTES = PRINTER_WIDTH_PIXELS // 8
MAX_FEED_CHUNK_HEIGHT = 256
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
DELAY_BETWEEN_PRINTS = 1.0
DEFAULT_FONT_NAME = "Arial"
DEFAULT_FONT_SIZE = 24

HD_SEQ_ENTER = [
    bytes.fromhex("0200020f000b000400520a002221a70000000000"),
    bytes.fromhex("02000210000c000400520a002221b10001000000ff"),
    bytes.fromhex("02000210000c000400520a002221a10001000000ff"),
    bytes.fromhex("02000210000c000400520a002221a2000100643bff"),
    bytes.fromhex("02000210000c000400520a002221a2000100643bff"),
]
HD_SEQ_EXIT = bytes.fromhex("02000210000c000400520a002221ad000100000000")

received_responses: Dict[int, bytes] = {}
notification_condition = asyncio.Condition()


def crc8_update(crc: int, data_byte: int) -> int:
    crc ^= data_byte
    for _ in range(8):
        if crc & 0x80:
            crc = (crc << 1) ^ 0x07
        else:
            crc <<= 1
        crc &= 0xFF
    return crc


def calculate_crc8(data: bytes) -> int:
    crc = 0x00
    for byte in data:
        crc = crc8_update(crc, byte)
    return crc


def create_command_with_crc(command_id: int, data: bytes) -> bytes:
    data_length_le = len(data).to_bytes(2, byteorder="little")
    crc = calculate_crc8(data)
    return bytes([0x22, 0x21, command_id, 0x00]) + data_length_le + data + bytes([crc, 0xFF])


def create_command_simple(command_id: int, data: bytes) -> bytes:
    data_length_le = len(data).to_bytes(2, byteorder="little")
    return bytes([0x22, 0x21, command_id, 0x00]) + data_length_le + data + bytes([0x00, 0x00])


async def write_chunked(client: BleakClient, char_uuid: str, data: bytes, max_chunk: int = 20, pause: float = 0.0) -> None:
    for index in range(0, len(data), max_chunk):
        await client.write_gatt_char(char_uuid, data[index:index + max_chunk], response=False)
        if pause:
            await asyncio.sleep(pause)


async def enter_hd_mode(client: BleakClient, data_char_uuid: str) -> None:
    for packet in HD_SEQ_ENTER:
        await client.write_gatt_char(data_char_uuid, packet, response=False)
        await asyncio.sleep(0.04)


async def exit_hd_mode(client: BleakClient, data_char_uuid: str) -> None:
    await client.write_gatt_char(data_char_uuid, HD_SEQ_EXIT, response=False)
    await asyncio.sleep(0.04)


def pack_hd_gray_line(line_pixels: Sequence[int]) -> bytes:
    values = [(pixel * 15) // 255 for pixel in line_pixels]
    packed = bytearray(
        ((values[i] << 4) | (values[i + 1] if i + 1 < len(values) else 0))
        for i in range(0, len(values), 2)
    )
    if len(packed) < 192:
        packed.extend(b"\x00" * (192 - len(packed)))
    return bytes(packed[:192])


def load_font(font_name: str, font_size: int) -> ImageFont.FreeTypeFont:
    font_path = None

    if MATPLOTLIB_AVAILABLE:
        matches = []
        try:
            for entry in font_manager.fontManager.ttflist:
                if entry.name == font_name:
                    matches.append(entry)
        except Exception:
            matches = []

        if matches:
            font_path = matches[0].fname
            indicators = ["bd", "bold", "bld", "i", "italic", "obl", "oblique", "blk", "black", "narrow"]
            for entry in matches:
                base = os.path.splitext(os.path.basename(entry.fname).lower())[0]
                remainder = base.replace(font_name.lower(), "")
                if not any(indicator in remainder for indicator in indicators):
                    font_path = entry.fname
                    break

        if not font_path:
            try:
                font_path = font_manager.findfont(font_name, fallback_to_default=False)
            except Exception:
                font_path = None

    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            pass

    if MATPLOTLIB_AVAILABLE:
        try:
            fallback = font_manager.findfont(DEFAULT_FONT_NAME, fallback_to_default=True)
            return ImageFont.truetype(fallback, font_size)
        except Exception:
            pass

    return ImageFont.load_default()


def otsu_threshold(image_l: Image.Image) -> int:
    hist = image_l.histogram()
    total = sum(hist)
    sum_total = sum(i * hist[i] for i in range(256))
    sum_background = 0.0
    weight_background = 0.0
    max_variance = -1.0
    threshold = 127

    for tone in range(256):
        weight_background += hist[tone]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += tone * hist[tone]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = tone
    return threshold


def ordered_dither(image_l: Image.Image, size: int = 4, gamma: float = 1.0, bias: int = 0) -> Image.Image:
    b4 = [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5],
    ]
    b8 = [
        [0, 48, 12, 60, 3, 51, 15, 63],
        [32, 16, 44, 28, 35, 19, 47, 31],
        [8, 56, 4, 52, 11, 59, 7, 55],
        [40, 24, 36, 20, 43, 27, 39, 23],
        [2, 50, 14, 62, 1, 49, 13, 61],
        [34, 18, 46, 30, 33, 17, 45, 29],
        [10, 58, 6, 54, 9, 57, 5, 53],
        [42, 26, 38, 22, 41, 25, 37, 21],
    ]

    source = image_l.point([int((i / 255.0) ** gamma * 255 + 0.5) for i in range(256)], "L") if abs(gamma - 1.0) > 1e-3 else image_l
    width, height = source.size
    out = Image.new("1", (width, height), 1)
    src = source.load()
    dst = out.load()

    if size == 4:
        scale = 256.0 / 17.0
        for y in range(height):
            row = b4[y & 3]
            for x in range(width):
                threshold = int(row[x & 3] * scale + bias)
                dst[x, y] = 0 if src[x, y] < threshold else 1
        return out

    if size == 8:
        mapped = [
            [min(255, max(0, int((b8[yy][xx] + 0.5) * (255.0 / 64.0) + bias))) for xx in range(8)]
            for yy in range(8)
        ]
        for y in range(height):
            row = mapped[y & 7]
            for x in range(width):
                dst[x, y] = 0 if src[x, y] < row[x & 7] else 1
        return out

    raise ValueError("size must be 4 or 8")


def ordered_dither_bayer2(image_l: Image.Image) -> Image.Image:
    matrix = ((0, 2), (3, 1))
    width, height = image_l.size
    src = image_l.load()
    out = Image.new("1", (width, height), 255)
    dst = out.load()

    for y in range(height):
        for x in range(width):
            threshold = ((matrix[y & 1][x & 1] + 0.5) / 4.0) * 255.0
            dst[x, y] = 0 if src[x, y] < threshold else 255
    return out


def floyd_steinberg_dense(image_l: Image.Image, gamma: float = 0.85) -> Image.Image:
    lut = [int(round(255 * ((i / 255.0) ** gamma))) for i in range(256)]
    return image_l.point(lut).convert("1", dither=Image.FLOYDSTEINBERG)


DIFFUSE_KERNELS = {
    "fs": {"div": 16, "weights": [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)]},
    "jarvis": {"div": 48, "weights": [(1, 0, 7), (2, 0, 5), (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3), (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)]},
    "stucki": {"div": 42, "weights": [(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2), (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)]},
    "burkes": {"div": 32, "weights": [(1, 0, 8), (2, 0, 4), (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)]},
    "sierra": {"div": 32, "weights": [(1, 0, 5), (2, 0, 3), (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2), (-1, 2, 2), (0, 2, 3), (1, 2, 2)]},
    "sierra2": {"div": 32, "weights": [(1, 0, 4), (2, 0, 3), (-2, 1, 1), (-1, 1, 2), (0, 1, 3), (1, 1, 2), (2, 1, 1)]},
    "sierra-lite": {"div": 4, "weights": [(1, 0, 2), (-1, 1, 1), (0, 1, 1)]},
    "atkinson": {"div": 8, "weights": [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)]},
}


def error_diffusion(image_l: Image.Image, kernel: Dict[str, Any], serpentine: bool = True) -> Image.Image:
    width, height = image_l.size
    src = image_l.copy().load()
    out = Image.new("1", (width, height), 1)
    dst = out.load()
    weights = kernel["weights"]
    divisor = kernel["div"]

    for y in range(height):
        x_range = range(width) if not (serpentine and (y % 2)) else range(width - 1, -1, -1)
        for x in x_range:
            old = src[x, y]
            new = 255 if old >= 128 else 0
            dst[x, y] = 1 if new == 255 else 0
            error = old - new

            for dx, dy, weight in weights:
                nx = x + (-dx if (serpentine and (y % 2)) else dx)
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    value = src[nx, ny] + (error * weight) / divisor
                    src[nx, ny] = int(max(0, min(255, value)) + 0.5)
    return out


def despeckle_isolated_black(img_1bit: Image.Image, gray_src: Image.Image, gray_thr: int = 246, neighbor_min: int = 2) -> Image.Image:
    width, height = img_1bit.size
    src = img_1bit.load()
    gray = gray_src.load()
    out = Image.new("1", (width, height), 1)
    dst = out.load()

    for y in range(height):
        for x in range(width):
            if src[x, y] != 0:
                dst[x, y] = 1
                continue
            if gray[x, y] < gray_thr:
                dst[x, y] = 0
                continue
            neighbors = 0
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    if (xx, yy) != (x, y) and src[xx, yy] == 0:
                        neighbors += 1
            dst[x, y] = 0 if neighbors >= neighbor_min else 1
    return out


def to_1bit(image_l: Image.Image, dither_mode: str, threshold_opt: str) -> Image.Image:
    mode = (dither_mode or "fs").lower()

    if mode == "dense":
        return ordered_dither_bayer2(image_l)
    if mode in ("fs", "floyd", "floyd-steinberg"):
        return floyd_steinberg_dense(image_l, gamma=0.85)
    if mode == "bayer4":
        return ordered_dither(image_l, size=4)
    if mode == "bayer8":
        mean = image_l.filter(ImageFilter.BoxBlur(2))
        dev = ImageChops.difference(image_l, mean).filter(ImageFilter.BoxBlur(2))
        white_mask = Image.eval(mean, lambda p: 255 if p >= 245 else 0).convert("1")
        flat_mask = Image.eval(dev, lambda p: 255 if p <= 4 else 0).convert("1")
        keep_white = ImageChops.logical_and(white_mask, flat_mask)
        dithered = ordered_dither(image_l, size=8, gamma=1.0, bias=0)
        dithered = Image.composite(Image.new("1", image_l.size, 1), dithered, keep_white)
        return despeckle_isolated_black(dithered, image_l, gray_thr=246, neighbor_min=2)
    if mode == "none":
        threshold = otsu_threshold(image_l) if str(threshold_opt).lower() == "auto" else max(0, min(255, int(threshold_opt)))
        return image_l.point(lambda p: 255 if p >= threshold else 0, mode="1")
    if mode in DIFFUSE_KERNELS:
        return error_diffusion(image_l, DIFFUSE_KERNELS[mode], serpentine=True)
    return image_l.convert("1", dither=Image.FLOYDSTEINBERG)


def prepare_image_for_print(image_path, target_width, dither, threshold, brightness):
    img = Image.open(image_path).convert("L")
    if img.width != target_width:
        ratio = target_width / img.width
        img = img.resize((target_width, int(round(img.height * ratio))), Image.LANCZOS)

    mode = (dither or "fs").lower()
    if mode == "none":
        lut = [int(round(255 * ((i / 255.0) ** 1.25))) for i in range(256)]
        gray = img.point(lut, "L")
        gray = gray.filter(ImageFilter.GaussianBlur(0.7))
        return gray

    gray = ImageOps.autocontrast(img, cutoff=1)
    brightness = max(0, min(255, int(brightness)))
    shift = brightness - 128
    img = img.point(lambda p: max(0, min(255, p + shift)), "L")
    gray = gray.filter(ImageFilter.UnsharpMask(radius=0.6, percent=120, threshold=2))
    return to_1bit(gray, dither, threshold)


def create_text_bitmap(text: str, font: ImageFont.ImageFont, width: int, alignment: str = "left") -> Image.Image:
    lines: List[str] = []
    paragraphs = text.split("\n")

    def text_width(value: str) -> int:
        if not value:
            return 0
        try:
            return int(font.getlength(value))
        except AttributeError:
            try:
                return font.getbbox(value)[2] - font.getbbox(value)[0]
            except AttributeError:
                return font.getsize(value)[0]

    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = current + (" " if current else "") + word
            if text_width(candidate) <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

    if not lines:
        return Image.new("1", (width, 1), color=1)

    try:
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        line_step = line_height + 4
        total_height = len(lines) * line_step - 4
    except AttributeError:
        line_height = font.getsize("A")[1]
        line_step = line_height + 2
        total_height = len(lines) * line_step - 2

    total_height = max(total_height, line_height)
    image = Image.new("1", (width, total_height), color=1)
    draw = ImageDraw.Draw(image)

    y = 0
    for line in lines:
        line_w = text_width(line)
        if alignment == "center":
            x = (width - line_w) // 2
        elif alignment == "right":
            x = width - line_w
        else:
            x = 0
        draw.text((x, y), line, font=font, fill=0)
        y += line_step

    return image


def process_image(image: Image.Image, overstrike: int = 1) -> bytes:
    pixels = image.load()
    width, height = image.size
    output = bytearray()

    def pack_row(y: int) -> bytearray:
        byte = 0
        bit_index = 0
        row = bytearray()
        for x in range(width):
            if pixels[x, y] == 0:
                byte |= (1 << (bit_index & 7))
            bit_index += 1
            if (bit_index & 7) == 0:
                row.append(byte)
                byte = 0
        if bit_index & 7:
            row.append(byte)
        return row

    repeat = max(1, min(3, int(overstrike)))
    for y in range(height):
        row = pack_row(y)
        for _ in range(repeat):
            output.extend(row)
    return bytes(output)


def parse_response(data: bytes) -> Tuple[Optional[int], Optional[bytes]]:
    if not data or len(data) < 8 or data[0] != 0x22 or data[1] != 0x21:
        return None, None
    command_id = data[2]
    payload_len = int.from_bytes(data[4:6], "little")
    if len(data) < 6 + payload_len:
        return command_id, None
    return command_id, data[6:6 + payload_len]


async def notification_handler(_: Any, data: bytes) -> None:
    command_id, payload = parse_response(data)
    if command_id is not None:
        async with notification_condition:
            received_responses[command_id] = payload or b""
            notification_condition.notify_all()


async def wait_for_response(command_id: int, timeout: float) -> bytes:
    async with notification_condition:
        received_responses.pop(command_id, None)
        await asyncio.wait_for(notification_condition.wait_for(lambda: command_id in received_responses), timeout=timeout)
        return received_responses.pop(command_id)


def check_a1_status(payload: bytes) -> bool:
    return bool(payload) and len(payload) >= 8 and payload[6] == 0


def check_a9_status(payload: bytes) -> bool:
    return bool(payload) and len(payload) >= 1 and payload[0] == 0


async def resolve_ble_device(device_addr: Optional[str], device_name: Optional[str], attempts: int = 2, scan_timeout: float = 6.0) -> Union[str, BLEDevice]:
    if not device_name:
        if not device_addr:
            raise RuntimeError("No device address or name provided.")
        return device_addr

    target = device_name.lower()
    adv_seen: Dict[str, Dict[str, Any]] = {}

    def name_match(device: BLEDevice, adv_data: Any = None) -> bool:
        if adv_data is not None:
            adv_seen[device.address] = {
                "local_name": getattr(adv_data, "local_name", None),
                "rssi": getattr(adv_data, "rssi", None),
            }
        name = getattr(adv_data, "local_name", None) if adv_data is not None else None
        if not name:
            name = getattr(device, "name", None)
        name = (name or "").lower()
        return name == target or name.startswith(target)

    def describe(device: BLEDevice) -> str:
        info = adv_seen.get(device.address, {})
        name = info.get("local_name") or getattr(device, "name", None) or ""
        rssi = info.get("rssi", getattr(device, "rssi", None))
        suffix = f" rssi={rssi}" if rssi is not None else ""
        return f"name='{name}' address='{device.address}'{suffix}"

    if hasattr(BleakScanner, "find_device_by_filter"):
        for attempt in range(1, attempts + 1):
            print(f"Scanning for '{device_name}' (attempt {attempt}/{attempts})...")
            device = await BleakScanner.find_device_by_filter(name_match, timeout=scan_timeout)
            if device:
                print("Found device:", describe(device))
                return device
            if attempt < attempts:
                await asyncio.sleep(1.0)

    for attempt in range(1, attempts + 1):
        print(f"Discovering devices (attempt {attempt}/{attempts})...")
        devices = await BleakScanner.discover(timeout=scan_timeout)
        for device in devices:
            if name_match(device):
                print("Found device:", describe(device))
                return device
        if attempt < attempts:
            await asyncio.sleep(1.0)

    raise RuntimeError(f"Could not find a BLE device named '{device_name}'.")


async def run_print_job(client, image, description, dither, overstrike, intensity):
    print(f"Processing: '{description}'")
    if image is None:
        print("Error: No image data provided.")
        return False

    is_hd = (dither or "").lower() == "none"
    width_bytes = PRINTER_WIDTH_BYTES

    if is_hd:
        image_height = image.height
        printer_data = None
        print(f"Prepared HD source: height={image_height} lines")
    else:
        printer_data = process_image(image, overstrike=overstrike)
        final_len = len(printer_data)
        if final_len % width_bytes != 0:
            print(f"Error: packed length {final_len} is not a multiple of {width_bytes} bytes.")
            return False
        image_height = final_len // width_bytes
        print(f"Prepared normal source: {final_len} bytes, height={image_height} lines")

    try:
        ae01 = client.services.get_characteristic(CONTROL_WRITE_UUID)
        ae02 = client.services.get_characteristic(NOTIFY_UUID)
        ae03 = client.services.get_characteristic(DATA_WRITE_UUID)
        if not all([ae01, ae02, ae03]):
            raise ValueError("Missing required GATT characteristic(s).")
    except Exception as exc:
        print(f"Error getting characteristics: {exc}")
        return False

    a2_value = max(0, min(255, int(intensity)))
    cmd_b1 = create_command_with_crc(0xB1, b"\x00")
    cmd_a2_1 = create_command_with_crc(0xA2, bytes([a2_value]))
    cmd_a1 = create_command_with_crc(0xA1, b"\x00")
    cmd_a2_2 = create_command_with_crc(0xA2, bytes([a2_value]))
    cmd_a9 = create_command_simple(0xA9, image_height.to_bytes(2, "little") + width_bytes.to_bytes(2, "little"))
    cmd_ad = create_command_simple(0xAD, b"\x00")

    try:
        for packet in (cmd_b1, cmd_a2_1, cmd_a1):
            await client.write_gatt_char(ae01.uuid, packet, response=False)
            await asyncio.sleep(0.01)
        if not check_a1_status(await wait_for_response(0xA1, 7.0)):
            raise RuntimeError("A1 status check failed.")

        if is_hd:
            print("Activating HD mode...")
            await enter_hd_mode(client, ae03.uuid)

        for packet in (cmd_a2_2, cmd_a9):
            await client.write_gatt_char(ae01.uuid, packet, response=False)
            await asyncio.sleep(0.01)
        if not check_a9_status(await wait_for_response(0xA9, 7.0)):
            raise RuntimeError("A9 status check failed.")

        if is_hd:
            print("Sending image data in HD mode...")
            gray = image if image.mode == "L" else image.convert("L")
            pixels = list(gray.getdata())
            for y in range(gray.height):
                line = pixels[y * gray.width:(y + 1) * gray.width]
                row = pack_hd_gray_line(line)
                await write_chunked(client, ae03.uuid, row, max_chunk=20, pause=0.0)
                await asyncio.sleep(0.005)
        else:
            print("Sending image data in normal mode...")
            await write_chunked(client, ae03.uuid, printer_data or b"", max_chunk=20, pause=0.0)
            await asyncio.sleep(0.30)

        await client.write_gatt_char(ae01.uuid, cmd_ad, response=False)
        await asyncio.sleep(0.01)

        if is_hd:
            print("Deactivating HD mode...")
            await exit_hd_mode(client, ae03.uuid)

        try:
            await wait_for_response(0xAA, max(15.0, image_height / 18.0))
            print("AA notification received.")
        except asyncio.TimeoutError:
            print("Warning: AA notification not received before timeout.")

        await asyncio.sleep(0.5)
        print(f"Print job for '{description}' completed.")
        return True
    except Exception as exc:
        print(f"Error during print job for '{description}': {exc}")
        return False


async def run_feed_paper(client: BleakClient, lines_to_feed: int) -> bool:
    if lines_to_feed <= 0:
        print("Error: lines to feed must be positive.")
        return False

    try:
        ae01 = client.services.get_characteristic(CONTROL_WRITE_UUID)
        ae02 = client.services.get_characteristic(NOTIFY_UUID)
        ae03 = client.services.get_characteristic(DATA_WRITE_UUID)
        if not all([ae01, ae02, ae03]):
            raise ValueError("Missing required GATT characteristic(s).")
    except Exception as exc:
        print(f"Error getting characteristics: {exc}")
        return False

    remaining = lines_to_feed
    chunk_index = 0

    while remaining > 0:
        chunk_index += 1
        current_lines = min(remaining, MAX_FEED_CHUNK_HEIGHT)
        print(f"Feed chunk {chunk_index}: {current_lines} lines")
        blank = bytes([0x00] * (PRINTER_WIDTH_BYTES * current_lines))

        cmd_b1 = create_command_with_crc(0xB1, b"\x00")
        cmd_a2_1 = create_command_with_crc(0xA2, bytes([0x5D]))
        cmd_a1 = create_command_with_crc(0xA1, b"\x00")
        cmd_a2_2 = create_command_with_crc(0xA2, bytes([0x5D]))
        cmd_a9 = create_command_simple(0xA9, current_lines.to_bytes(2, "little") + PRINTER_WIDTH_BYTES.to_bytes(2, "little"))
        cmd_ad = create_command_simple(0xAD, b"\x00")

        try:
            for packet in (cmd_b1, cmd_a2_1, cmd_a1):
                await client.write_gatt_char(ae01.uuid, packet, response=False)
                await asyncio.sleep(0.01)
            if not check_a1_status(await wait_for_response(0xA1, 7.0)):
                raise RuntimeError("A1 status check failed.")

            for packet in (cmd_a2_2, cmd_a9):
                await client.write_gatt_char(ae01.uuid, packet, response=False)
                await asyncio.sleep(0.01)
            if not check_a9_status(await wait_for_response(0xA9, 7.0)):
                raise RuntimeError("A9 status check failed.")

            await write_chunked(client, ae03.uuid, blank, max_chunk=20, pause=0.0)
            await client.write_gatt_char(ae01.uuid, cmd_ad, response=False)
            await asyncio.sleep(0.01)
            await wait_for_response(0xAA, max(15.0, current_lines / 20.0))
            await asyncio.sleep(0.5)
        except Exception as exc:
            print(f"Error during feed chunk {chunk_index}: {exc}")
            return False

        remaining -= current_lines
        if remaining > 0:
            await asyncio.sleep(DELAY_BETWEEN_PRINTS)

    print(f"Feed sequence completed: {lines_to_feed} lines")
    return True


def validate_args(args: argparse.Namespace) -> bool:
    has_print_content = bool(args.image or args.folder or args.text)
    has_feed = args.feed is not None
    action_count = sum([has_print_content, has_feed])

    if action_count == 0:
        print("Error: no action specified. Use -i, -f, -t, or -p.")
        return False
    if has_feed and has_print_content:
        print("Error: --feed cannot be combined with image/folder/text actions.")
        return False
    return True


def prepare_print_jobs(args: argparse.Namespace) -> List[Tuple[Image.Image, str]]:
    jobs: List[Tuple[Image.Image, str]] = []

    if args.image:
        image = prepare_image_for_print(args.image, PRINTER_WIDTH_PIXELS, args.dither, args.threshold, args.brightness)
        if args.upside_down:
            image = image.rotate(180)
        jobs.append((image, os.path.basename(args.image)))

    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"Error: folder not found: {args.folder}")
        else:
            for filename in sorted(os.listdir(args.folder)):
                if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
                    continue
                path = os.path.join(args.folder, filename)
                image = prepare_image_for_print(path, PRINTER_WIDTH_PIXELS, args.dither, args.threshold, args.brightness)
                if args.upside_down:
                    image = image.rotate(180)
                jobs.append((image, filename))

    if args.text:
        font = load_font(args.font, args.font_size)
        image = create_text_bitmap(args.text, font, PRINTER_WIDTH_PIXELS, alignment=args.align)
        if args.upside_down:
            image = image.rotate(180)
        jobs.append((image, f"Text: {args.text[:32]}"))

    return jobs


def save_debug_images(jobs: Sequence[Tuple[Image.Image, str]], upside_down: bool) -> int:
    output_dir = Path("debug_output")
    output_dir.mkdir(exist_ok=True)

    for index, (image, desc) in enumerate(jobs):
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in desc)
        safe = safe.replace(" ", "_").strip("._-")[:40] or f"job_{index:02d}"
        suffix = "_rotated" if upside_down else ""
        path = output_dir / f"debug_{index:02d}_{safe}{suffix}.png"
        image.save(path)
        print(f"Saved: {path}")
    return 0


def list_system_fonts() -> int:
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib is not installed. Install it with: pip install matplotlib")
        return 1
    fonts = sorted(set(entry.name for entry in font_manager.fontManager.ttflist))
    for font_name in fonts:
        print(font_name)
    return 0


async def run(args: argparse.Namespace) -> int:
    if args.list_fonts:
        return list_system_fonts()

    if not validate_args(args):
        return 1

    try:
        target = await resolve_ble_device(args.device, args.device_name)
    except Exception as exc:
        print(f"Error resolving device: {exc}")
        return 1

    if args.feed is not None:
        print(f"Attempting to connect to printer at {target}...")
        try:
            async with BleakClient(target, timeout=45.0) as client:
                if not client.is_connected:
                    print("Failed to connect.")
                    return 1
                print(f"Connected to {client.address}")
                await client.start_notify(NOTIFY_UUID, notification_handler)
                try:
                    success = await run_feed_paper(client, args.feed)
                finally:
                    await asyncio.sleep(1.0)
                    await client.stop_notify(NOTIFY_UUID)
                return 0 if success else 1
        except BleakError as exc:
            print(f"Bluetooth error: {exc}")
            return 1
        except asyncio.TimeoutError:
            print("Connection timed out.")
            return 1

    jobs = prepare_print_jobs(args)
    if not jobs:
        print("Error: no valid print jobs could be prepared.")
        return 1

    if args.debug_save:
        return save_debug_images(jobs, args.upside_down)

    print(f"Prepared {len(jobs)} print job(s).")
    try:
        async with BleakClient(target, timeout=45.0) as client:
            if not client.is_connected:
                print("Failed to connect.")
                return 1
            print(f"Connected to {client.address}")
            await client.start_notify(NOTIFY_UUID, notification_handler)
            try:
                all_ok = True
                for index, (image, desc) in enumerate(jobs, start=1):
                    print(f"--- Print job {index}/{len(jobs)} ---")
                    ok = await run_print_job(client, image, desc, args.dither, args.overstrike, args.intensity)
                    all_ok = all_ok and ok
                    if index < len(jobs):
                        await asyncio.sleep(DELAY_BETWEEN_PRINTS)
            finally:
                await asyncio.sleep(1.0)
                await client.stop_notify(NOTIFY_UUID)
            return 0 if all_ok else 1
    except BleakError as exc:
        print(f"Bluetooth error: {exc}")
        return 1
    except asyncio.TimeoutError:
        print("Connection timed out.")
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean MXW01 thermal printer CLI")
    parser.add_argument("-i", "--image", help="Path to a single image file to print.")
    parser.add_argument("-f", "--folder", help="Path to a folder containing images to print.")
    parser.add_argument("-t", "--text", help="Text string to print.")
    parser.add_argument("-p", "--feed", type=int, default=None, const=40, nargs="?", help="Feed paper by the specified number of lines.")
    parser.add_argument("-d", "--device", default=DEFAULT_ADDRESS, help="Bluetooth MAC address of the printer.")
    parser.add_argument("-N", "--device-name", help="Bluetooth device name; if set, scanning is used and overrides --device.")
    parser.add_argument("-s", "--debug-save", action="store_true", help="Save prepared bitmaps to debug_output instead of printing.")
    parser.add_argument("-u", "--upside-down", action="store_true", help="Rotate images/text 180 degrees before printing.")
    parser.add_argument("-z", "--font-size", type=int, default=DEFAULT_FONT_SIZE, help="Font size for text printing.")
    parser.add_argument("-n", "--font", default=DEFAULT_FONT_NAME, help="Font name for text printing.")
    parser.add_argument("-a", "--align", choices=["left", "center", "right"], default="left", help="Text alignment.")
    parser.add_argument("-l", "--list-fonts", action="store_true", help="List available system fonts and exit.")
    parser.add_argument(
        "--dither",
        choices=["none", "fs", "atkinson", "jarvis", "stucki", "burkes", "sierra", "sierra2", "sierra-lite", "bayer4", "bayer8", "dense"],
        default="fs",
        help="Image conversion mode.",
    )
    parser.add_argument("--overstrike", type=int, default=1, help="Repeat each raster line 1..3 times to increase darkness.")
    parser.add_argument("--threshold", default="auto", help="0..255 or 'auto' when using non-dither thresholding.")
    parser.add_argument("--brightness", type=int, default=128,
                    help="Image brightness 0..255. Lower is darker, higher is lighter.")
    parser.add_argument("--intensity", type=int, default=93,
                    help="Printer intensity 0..255. Default: 93.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())