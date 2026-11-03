"""Convert images, animated GIFs, and videos to block-based pixel art.

Run from a terminal:
    python pixel_art.py photo.png
    python pixel_art.py animation.gif --block-size 10
    python pixel_art.py video.mp4 --output result.mp4

Output files are saved to disk; no preview window is opened.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence


# Default settings for the edge-aware pixelation algorithm.
BLOCK_SIZE = 15
DARK_THRESHOLD = 60
MIN_DARK_FRACTION = 0.30
MIN_EDGE_FRACTION = 0.05
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def divide_into_blocks(image, block_size):
    """Group image pixels into non-overlapping blocks for calculations.

    The right and bottom borders are padded with the nearest pixel values.
    Returned arrays have shape (block_rows, block_columns, pixels_per_block)
    or (block_rows, block_columns, pixels_per_block, channels).
    """
    height, width = image.shape[:2]
    bottom_padding = (-height) % block_size
    right_padding = (-width) % block_size

    padding = [(0, bottom_padding), (0, right_padding)]
    if image.ndim == 3:
        padding.append((0, 0))  # Do not pad the color channels.

    padded_image = np.pad(image, padding, mode="edge")
    padded_height, padded_width = padded_image.shape[:2]
    block_rows = padded_height // block_size
    block_columns = padded_width // block_size
    pixels_per_block = block_size * block_size

    if image.ndim == 3:
        channels = image.shape[2]
        blocks = padded_image.reshape(
            block_rows, block_size, block_columns, block_size, channels
        )
        blocks = blocks.transpose(0, 2, 1, 3, 4)
        return blocks.reshape(block_rows, block_columns, pixels_per_block, channels)

    blocks = padded_image.reshape(block_rows, block_size, block_columns, block_size)
    blocks = blocks.transpose(0, 2, 1, 3)
    return blocks.reshape(block_rows, block_columns, pixels_per_block)


def pixelate_image(
    image,
    block_size=BLOCK_SIZE,
    dark_threshold=DARK_THRESHOLD,
    min_dark_fraction=MIN_DARK_FRACTION,
    min_edge_fraction=MIN_EDGE_FRACTION,
):
    """Use median colors for fills and dark-pixel means for eligible edge blocks.

    Input and output images have the same dimensions and color-channel order.
    """
    if block_size < 1:
        raise ValueError("Block size must be at least 1.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("The input must be a three-channel color image.")

    height, width = image.shape[:2]

    # Smooth intensity before applying Canny edge detection.
    grayscale = image.mean(axis=2).astype(np.uint8)
    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    detected_edges = cv2.Canny(
        blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD
    ) > 0
    edge_blocks = divide_into_blocks(detected_edges, block_size)
    edge_fraction = edge_blocks.mean(axis=2)

    # Determine which pixels in each block count as dark.
    color_blocks = divide_into_blocks(image, block_size)
    pixel_intensity = color_blocks.mean(axis=3)
    dark_pixel_mask = pixel_intensity < dark_threshold
    dark_fraction = dark_pixel_mask.mean(axis=2)

    # Compute the ordinary median and the mean of only dark pixels.
    median_colors = np.median(color_blocks, axis=2)
    number_of_dark_pixels = dark_pixel_mask.sum(axis=2)
    dark_color_sums = (color_blocks * dark_pixel_mask[..., None]).sum(axis=2)
    safe_dark_counts = np.maximum(number_of_dark_pixels, 1)
    dark_mean_colors = dark_color_sums / safe_dark_counts[..., None]

    # Preserve dark lines only when both outline conditions are met.
    use_dark_color = (
        (dark_fraction >= min_dark_fraction)
        & (edge_fraction >= min_edge_fraction)
    )
    block_colors = np.where(
        use_dark_color[..., None], dark_mean_colors, median_colors
    )

    # Expand each block color into a square, then remove border padding.
    result = np.repeat(block_colors, block_size, axis=0)
    result = np.repeat(result, block_size, axis=1)
    return result[:height, :width].astype(np.uint8)


def process_image(input_path, output_path, settings):
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    result = pixelate_image(image, **settings)
    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Could not save image: {output_path}")


def process_gif(input_path, output_path, settings):
    processed_frames = []
    frame_durations = []

    with Image.open(input_path) as animation:
        repeat_count = animation.info.get("loop", 0)
        default_duration = animation.info.get("duration", 100)

        for frame in ImageSequence.Iterator(animation):
            rgb_frame = np.asarray(frame.convert("RGB"))
            pixelated_frame = pixelate_image(rgb_frame, **settings)
            processed_frames.append(Image.fromarray(pixelated_frame))
            frame_durations.append(frame.info.get("duration", default_duration))

    if not processed_frames:
        raise ValueError("The GIF contains no frames.")

    processed_frames[0].save(
        output_path,
        save_all=True,
        append_images=processed_frames[1:],
        duration=frame_durations,
        loop=repeat_count,
        disposal=2,
    )


def process_video(input_path, output_path, settings):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FFmpeg is required for video processing.")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    frames_per_second = capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frames_per_second <= 0 or frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise ValueError("Could not determine the video dimensions or frame rate.")

    with tempfile.TemporaryDirectory() as temporary_folder:
        temporary_video = Path(temporary_folder) / "processed_frames.mp4"
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(temporary_video), codec, frames_per_second,
            (frame_width, frame_height)
        )
        if not writer.isOpened():
            capture.release()
            raise OSError("Could not create the temporary video.")

        frame_count = 0
        try:
            while True:
                has_frame, frame = capture.read()
                if not has_frame:
                    break
                writer.write(pixelate_image(frame, **settings))
                frame_count += 1
        finally:
            capture.release()
            writer.release()

        if frame_count == 0:
            raise ValueError("The video contains no readable frames.")

        # Encode the processed frames as H.264. Audio is not retained.
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(temporary_video),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(output_path),
            ],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert an image, GIF, or video into edge-aware pixel art."
    )
    parser.add_argument("input", type=Path, help="Path to the original file")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output file path (default: <input>_pixelated.png/gif/mp4)",
    )
    parser.add_argument(
        "-k", "--block-size", type=int, default=BLOCK_SIZE,
        help=f"Block width and height in pixels (default: {BLOCK_SIZE})",
    )
    parser.add_argument("--dark-threshold", type=float, default=DARK_THRESHOLD)
    parser.add_argument("--min-dark-fraction", type=float, default=MIN_DARK_FRACTION)
    parser.add_argument("--min-edge-fraction", type=float, default=MIN_EDGE_FRACTION)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")
    if args.block_size < 1:
        parser.error("--block-size must be at least 1")
    if not 0 <= args.dark_threshold <= 255:
        parser.error("--dark-threshold must be between 0 and 255")
    if not 0 <= args.min_dark_fraction <= 1 or not 0 <= args.min_edge_fraction <= 1:
        parser.error("Fraction settings must be between 0 and 1")

    extension = args.input.suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        output_extension = extension
        processor = process_image
    elif extension == ".gif":
        output_extension = ".gif"
        processor = process_gif
    elif extension in VIDEO_EXTENSIONS:
        output_extension = ".mp4"
        processor = process_video
    else:
        parser.error(f"Unsupported input format: {extension}")

    output_path = args.output or args.input.with_name(
        f"{args.input.stem}_pixelated{output_extension}"
    )
    if output_path.suffix.lower() != output_extension:
        parser.error(f"The output file must use the {output_extension} extension")
    if output_path.resolve() == args.input.resolve():
        parser.error("The output file must not overwrite the input file")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    settings = {
        "block_size": args.block_size,
        "dark_threshold": args.dark_threshold,
        "min_dark_fraction": args.min_dark_fraction,
        "min_edge_fraction": args.min_edge_fraction,
    }
    processor(args.input, output_path, settings)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
