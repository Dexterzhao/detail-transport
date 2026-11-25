"""Draw the peak transport measurements and rest reliefs for a completed run."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pilot import ROOT, read_config, relief


def font(size, bold=False):
    candidates = ("calibrib.ttf", "DejaVuSans-Bold.ttf") if bold else ("calibri.ttf", "DejaVuSans.ttf")
    for filename in candidates:
        for path in (Path("C:/Windows/Fonts") / filename, Path(filename)):
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def read_peak(run_dir, config):
    peak_step = int(np.argmax(config["peak_sequence"]))
    peak = {}
    with (run_dir / "measures.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["step"]) != peak_step:
                continue
            key = (row["asset"], row["policy"])
            percent = 100 * float(row["detail_length_change_nrmse"])
            if key in peak or not math.isfinite(percent) or percent < 0:
                raise ValueError(f"Invalid peak measurement for {key}.")
            if float(row["amplitude"]) != config["peak_sequence"][peak_step]:
                raise ValueError(f"Peak amplitude disagrees with the run configuration for {key}.")
            peak[key] = percent
    expected = {(field, policy) for field in config["assets"] for policy in config["policies"]}
    if peak.keys() != expected:
        raise ValueError("Peak rows must match all field/policy pairs in the saved run configuration.")
    return peak


def draw_figure(run_dir, output):
    config = read_config(run_dir / "config.json")
    peak = read_peak(run_dir, config)
    styles = {
        ("rotation", 0.0): ("Rotation only", "#34414d"),
        ("half_stretch", 0.5): ("Square-root stretch", "#138477"),
        ("full_gradient", 1.0): ("Full gradient", "#bc6038"),
    }
    palette = ("#595391", "#33729a", "#77533c", "#8e4163")
    legend = []
    for i, (name, exponent) in enumerate(config["policies"].items()):
        label, color = styles.get((name, exponent), (f"{name} (p={exponent:g})", palette[i % len(palette)]))
        legend.append((name, label, color))

    ink, muted, grid_color = "#182b40", "#415567", "#d4dee4"
    label_font = font(27)
    x0 = 375
    width = max(1390, math.ceil(x0 + sum(label_font.getlength(label) + 48 for _, label, _ in legend)))
    row_height = max(125, 26 * len(legend) + 40)
    chart_height = row_height * len(config["assets"])
    axis_y = 111 + chart_height
    canvas = Image.new("RGB", (width, axis_y + 84), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((25, 10), "Rest relief", font=font(32, True), fill=ink)
    draw.text((355, 10), "Relative RMS change in displacement-vector length (%)", font=font(32, True), fill=ink)
    x = 355
    for _, label, color in legend:
        draw.ellipse((x, 67, x + 15, 82), fill=color)
        draw.text((x + 23, 57), label, font=label_font, fill=ink)
        x += math.ceil(label_font.getlength(label)) + 48

    max_percent = max(1.0, max(peak.values()))
    step = 10 ** math.floor(math.log10(max_percent / 8))
    for multiple in (1, 2, 5, 10):
        if math.ceil(max_percent / (step * multiple)) <= 10:
            step *= multiple
            break
    tick_count = math.ceil(max_percent / step)
    horizontal_scale = (width - x0 - 97) / (tick_count * step)
    for i in range(tick_count + 1):
        value = i * step
        tick_x = round(x0 + value * horizontal_scale)
        draw.line((tick_x, 110, tick_x, axis_y), fill=grid_color, width=2)
        draw.text((tick_x, axis_y + 5), f"{value:g}", font=font(25), fill=muted, anchor="mt")
    draw.line((x0, axis_y, width - 97, axis_y), fill=muted, width=2)

    axis = np.linspace(0, 1, 100)
    xx, yy = np.meshgrid(axis, axis[::-1])
    coordinates = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    light = np.array([243, 241, 233])
    dark = np.array([39, 65, 76])
    for k, asset in enumerate(config["assets"]):
        y = 103 + row_height // 2 + row_height * k
        intensity = np.clip(relief(coordinates, asset)[:, 2].reshape(xx.shape) / 0.055, 0, 1)
        field = np.uint8(np.round(light + (dark - light) * intensity[:, :, None]))
        canvas.paste(Image.fromarray(field), (28, y - 50))
        draw.rectangle((28, y - 50, 127, y + 49), outline="#9daeb3", width=1)
        draw.text((146, y), asset.replace("_", " ").capitalize(), font=font(27, True), fill=ink, anchor="lm")
        for i, (policy, _, color) in enumerate(legend):
            value = peak[asset, policy]
            end = round(x0 + value * horizontal_scale)
            line_y = y + 26 * (i - (len(legend) - 1) / 2)
            draw.line((x0, line_y, end, line_y), fill=color, width=4)
            draw.ellipse((end - 6, line_y - 6, end + 6, line_y + 6), fill=color)
            draw.text((end + 12, line_y), f"{value:.2f}", font=font(23), fill=ink, anchor="lm")
    draw.text((25, axis_y + 46), "Inset shading shows the rest normal offset on the same 0–0.055 patch-width scale.",
              fill=muted, font=font(26))
    canvas.save(output, format="PNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "pilot")
    args = parser.parse_args()
    directory = args.run_dir.resolve()
    if directory.is_relative_to(ROOT / "reference") or directory.is_relative_to(ROOT / ".git"):
        parser.error("Render into a generated run directory, outside the reference snapshot and Git metadata.")
    try:
        draw_figure(directory, directory / "transport_check.png")
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Cannot render figure: {error}\n")
    print(directory / "transport_check.png")


if __name__ == "__main__":
    main()
