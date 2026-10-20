"""
DRG tracer analysis – combined pipeline (tracer_only + 1810 + sum1 + 1810sum).

1. (Optional) Interactive MIP segmentation of .nd2 files, per channel:
   threshold slider, L-click add/restore, R-click delete, live diameter labels.
2. Upserts results into master CSVs (one representation per file_path + channel):
     drg_mip_outputs/COMBINED_master_counts.csv
     drg_mip_outputs/COMBINED_master_per_cell.csv
3. Summaries:
     - cell count per DRG level, split by channel   (counts_per_drg_level.csv)
     - soma diameter stats + fiber types per channel (diameter_per_channel.csv)
4. Plots:
     - cell count per DRG level per channel (totals + mean ± SEM of samples)
     - soma diameter distribution per channel + fiber-type counts per channel

Expected folder layout: <experiment>/<side Lt|Rt>/<level>.nd2  (e.g. 8.10.26/Rt/L5.nd2)
"""

from datetime import datetime
from pathlib import Path
import re

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nd2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, distance_transform_edt, gaussian_filter
from scipy.stats import gaussian_kde
import seaborn as sns
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, h_maxima, opening
from skimage.segmentation import find_boundaries, watershed

# ==========================================
# CONFIG
# ==========================================
DEFAULT_INPUT_PATH = "drg_samples"
CHANNELS = [1, 2]
CHANNEL_NAMES = {1: "Lymph Node", 2: "SC"}
CHANNEL_COLORS = {"Lymph Node": "#24588e", "SC": "#d65319"}

# סינון תאים (מיקרונים וצורה)
MIN_DIAM_UM = 16.0
MAX_DIAM_UM = 55.0
BORDER_MARGIN_PX = 20
MIN_CIRCULARITY = 0.22
MAX_ECCENTRICITY = 0.88

CLOSING_RADIUS_PX = 3
OPENING_RADIUS_PX = 1
MIN_SIZE_PX = 80
HIGHLIGHT_COLOR = np.array([0.0, 1.0, 1.0])

# פיצול תאים צמודים (watershed) לפי עמק בהירות כהה בין תאים + "צוואר" בצורה.
# SPLIT_DEPTH קטן יותר = מפצל יותר (סכנה לחתוך תא אחד לחלקים), גדול יותר = מפצל פחות.
SPLIT_TOUCHING_CELLS = True
SPLIT_SMOOTH_UM = 3.0
SPLIT_DEPTH = 0.10

# "תא בהיר" = בהיר יותר מהרקמה שסביבו, לא רק בהיר באופן מוחלט:
# מחסירים רקע מקומי (טשטוש בקנה מידה BACKGROUND_UM) והסף (הסליידר) מופעל על התמונה אחרי ההחסרה.
# BACKGROUND_UM = 0 מחזיר את השיטה הישנה (סף מוחלט).
BACKGROUND_UM = 25.0
BACKGROUND_DEFAULT_THRESH = 0.08
# כל תא חייב להיות בהיר לפחות ב-MIN_CONTRAST מטבעת של CONTRAST_RING_PX פיקסלים סביבו
MIN_CONTRAST = 0.10
CONTRAST_RING_PX = 6

# גבולות סיווג סיבים בסומות DRG (מיקרונים)
C_FIBER_MAX_DIAM = 25.0       # <= 25 um -> C-fiber
A_DELTA_MAX_DIAM = 35.0       # 25 < diam <= 35 um -> A-delta, > 35 um -> A-beta
FIBER_ORDER = ["C-fiber", "A-delta", "A-beta/Large"]
FIBER_COLORS = {"C-fiber": "#2b5c8f", "A-delta": "#d95f02", "A-beta/Large": "#7570b3"}

# סדר רמות ה-DRG בגרפים (רמות שלא ברשימה יתווספו בסוף)
DRG_ORDER = ["C", "T", "L1", "L2", "L3", "L4", "L5", "L6", "S", "S1", "S2", "S3"]

BASE_OUTPUT_DIR = Path("drg_mip_outputs")
MASTER_COUNTS_CSV = BASE_OUTPUT_DIR / "COMBINED_master_counts.csv"
MASTER_PER_CELL_CSV = BASE_OUTPUT_DIR / "COMBINED_master_per_cell.csv"

TIMESTAMP_STR = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_OUTPUT_DIR = BASE_OUTPUT_DIR / f"combined_run_{TIMESTAMP_STR}"


# ==========================================
# UTILS
# ==========================================
def classify_soma_size(diam):
    if diam <= C_FIBER_MAX_DIAM:
        return "C-fiber"
    if diam <= A_DELTA_MAX_DIAM:
        return "A-delta"
    return "A-beta/Large"


def extract_drg_level(fname):
    """L5.nd2 -> L5, L510.nd2 -> L5, L6_2.nd2 -> L6, L4Lt.nd2 -> L4, T_2.nd2 -> T, S_1.nd2 -> S."""
    m = re.match(r"^([A-Za-z])(\d?)", Path(str(fname)).stem)
    return f"{m.group(1).upper()}{m.group(2)}" if m else Path(str(fname)).stem


def drg_sort_key(level):
    return (DRG_ORDER.index(level), "") if level in DRG_ORDER else (len(DRG_ORDER), level)


def upsert_master(new_df, master_path, sort_cols):
    """מחליף את כל השורות של אותו file_path + channel בתוצאות ההרצה האחרונה (ללא כפילויות)."""
    master_path = Path(master_path)
    if master_path.exists() and master_path.stat().st_size > 0:
        master_df = pd.read_csv(master_path)
        new_keys = set(zip(new_df["file_path"], new_df["channel"]))
        keep = [(fp, ch) not in new_keys for fp, ch in zip(master_df["file_path"], master_df["channel"])]
        master_df = pd.concat([master_df[keep], new_df], ignore_index=True)
    else:
        master_df = new_df
    master_df = master_df.sort_values(by=sort_cols).reset_index(drop=True)
    master_df.to_csv(master_path, index=False, encoding="utf-8-sig")
    return master_df


def get_nd2_channel_color(f_nd2, ch_idx):
    """חילוץ צבע הערוץ המוגדר מתוך המטא-דאטה של קובץ ה-ND2."""
    try:
        channels_meta = f_nd2.metadata.channels
        if channels_meta and ch_idx < len(channels_meta):
            ch = channels_meta[ch_idx].channel
            c_obj = getattr(ch, "colorRGB", None) or getattr(ch, "color", None)
            if c_obj is not None:
                if hasattr(c_obj, "r") and hasattr(c_obj, "g") and hasattr(c_obj, "b"):
                    r, g, b = c_obj.r, c_obj.g, c_obj.b
                elif isinstance(c_obj, (list, tuple)) and len(c_obj) >= 3:
                    r, g, b = c_obj[0], c_obj[1], c_obj[2]
                else:
                    color_int = int(c_obj) & 0xFFFFFF
                    r, g, b = (color_int >> 16) & 0xFF, (color_int >> 8) & 0xFF, color_int & 0xFF
                if max(r, g, b) > 1.0:
                    r, g, b = r / 255.0, g / 255.0, b / 255.0
                if r == 0 and g == 0 and b == 0:
                    return np.array([1.0, 1.0, 1.0], dtype=np.float32)
                return np.array([r, g, b], dtype=np.float32)
    except Exception as e:
        print(f"Error reading channel {ch_idx} color: {e}")
    fallback_map = {0: [0.2, 0.5, 1.0], 1: [1.0, 0.1, 0.1], 2: [0.1, 1.0, 0.1]}
    return np.array(fallback_map.get(ch_idx, [1.0, 1.0, 1.0]), dtype=np.float32)


def calculate_default_threshold(norm_mip):
    """חישוב סף דינמי ראשוני על סמך רמת רעש הרקע."""
    tissue_fraction = np.mean(norm_mip > 0.10)
    if tissue_fraction < 0.15:
        return 0.12
    tissue_pixels = norm_mip[norm_mip > 0.05]
    if len(tissue_pixels) == 0:
        return 0.25
    computed = float(np.median(tissue_pixels) + 0.55 * np.std(tissue_pixels))
    return float(np.clip(computed, 0.22, 0.55))


def split_touching_cells(filled, intensity, pixel_size_um):
    """
    מפצל תאים צמודים (watershed). המשטח משלב בהירות מוחלקת (עמק כהה בין תאים)
    ומרחק מהשפה (צוואר בצורה, עד 8 מיקרון). רק פסגה בעומק SPLIT_DEPTH ומעלה הופכת לתא נפרד.
    """
    if not SPLIT_TOUCHING_CELLS:
        return label(filled, connectivity=2)
    smooth = gaussian_filter(intensity, SPLIT_SMOOTH_UM / pixel_size_um)
    dist_um = distance_transform_edt(filled) * pixel_size_um
    surface = np.where(filled, smooth + 0.5 * np.minimum(dist_um, 8.0) / 8.0, 0)
    markers = label(h_maxima(surface, SPLIT_DEPTH))
    return watershed(-surface, markers, mask=filled)


def force_split(mask, pixel_size_um):
    """פיצול אגרסיבי (מקש x): כל מרכז במרחק של לפחות ~רדיוס תא מינימלי מקבל תא משלו."""
    dist = gaussian_filter(distance_transform_edt(mask), 1.0)
    min_d = max(3, int(round(0.8 * (MIN_DIAM_UM / 2.0) / pixel_size_um)))
    peaks = peak_local_max(dist, min_distance=min_d, labels=mask.astype(int), exclude_border=False)
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    return watershed(-dist, markers, mask=mask)


# ==========================================
# INTERACTIVE SEGMENTATION
# ==========================================
def process_channel_mip(file_path, ch_idx, raw_stack, c_axis, pixel_area_um2, pixel_size_um, h_dim, w_dim, base_col):
    raw_tr = np.take(raw_stack, indices=ch_idx, axis=c_axis) if c_axis is not None else raw_stack
    mip_raw = raw_tr.max(axis=0).astype(np.float32)

    p_l, p_h = np.percentile(mip_raw, (2.0, 99.8))
    norm_mip = np.clip((mip_raw - p_l) / (p_h - p_l + 1e-5), 0, 1)

    # התמונה שעליה מופעל הסף: אחרי החסרת רקע מקומי (או norm_mip בשיטה הישנה)
    if BACKGROUND_UM > 0:
        thresh_img = norm_mip - gaussian_filter(norm_mip, BACKGROUND_UM / pixel_size_um)
        default_thresh = BACKGROUND_DEFAULT_THRESH
    else:
        thresh_img = norm_mip
        default_thresh = calculate_default_threshold(norm_mip)
    manual_search_radius_px = int(round((MAX_DIAM_UM / 2.0) / pixel_size_um))

    def local_contrast(coords, occupied):
        """בהירות ממוצעת בתא פחות בהירות ממוצעת בטבעת סביבו (בלי תאים שכנים)."""
        m = CONTRAST_RING_PX
        y0, x0 = np.maximum(coords.min(axis=0) - m, 0)
        y1, x1 = np.minimum(coords.max(axis=0) + m + 1, [h_dim, w_dim])
        cell = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        cell[coords[:, 0] - y0, coords[:, 1] - x0] = True
        near = distance_transform_edt(~cell) <= m
        ring = near & ~cell & ~occupied[y0:y1, x0:x1]
        if ring.sum() < 10:
            ring = near & ~cell
        crop = norm_mip[y0:y1, x0:x1]
        return float(crop[cell].mean() - crop[ring].mean())

    def diam_of(area_px):
        return 2.0 * np.sqrt((area_px * pixel_area_um2) / np.pi)

    def describe(prop, oy=0, ox=0, is_manual=False):
        """מדידות של תא מתוך regionprops (oy/ox = היסט של חיתוך מקומי). מחזיר (קואורדינטות, info)."""
        coords = prop.coords + np.array([oy, ox])
        cy, cx = prop.centroid
        return coords, {
            "cy": cy + oy, "cx": cx + ox,
            "diam_um": diam_of(prop.area),
            "major_diam_um": prop.major_axis_length * pixel_size_um,
            "minor_diam_um": prop.minor_axis_length * pixel_size_um,
            "area_um2": prop.area * pixel_area_um2,
            "brightness": float(np.percentile(norm_mip[coords[:, 0], coords[:, 1]], 90)),
            "circularity": (4.0 * np.pi * prop.area) / (prop.perimeter ** 2 + 1e-6),
            "eccentricity": prop.eccentricity,
            "is_manual": is_manual,
        }

    def segment_at_threshold(thresh_val):
        binary = thresh_img >= thresh_val
        if CLOSING_RADIUS_PX > 0:
            binary = closing(binary, disk(CLOSING_RADIUS_PX))
        if OPENING_RADIUS_PX > 0:
            binary = opening(binary, disk(OPENING_RADIUS_PX))
        # מילוי חלל הציטופלזמה והגרעין לקבלת שטח הסומה המלא, ואז פיצול תאים צמודים
        filled = binary_fill_holes(binary)
        m = split_touching_cells(filled, norm_mip, pixel_size_um)

        masks = np.zeros(m.shape, dtype=np.int32)
        info = {}
        idx = 1
        for prop in regionprops(m):
            if prop.area < MIN_SIZE_PX:
                continue
            cy, cx = prop.centroid
            if (cy < BORDER_MARGIN_PX or cy > h_dim - BORDER_MARGIN_PX or
                    cx < BORDER_MARGIN_PX or cx > w_dim - BORDER_MARGIN_PX):
                continue
            if not (MIN_DIAM_UM <= diam_of(prop.area) <= MAX_DIAM_UM):
                continue
            coords, cell = describe(prop)
            if cell["circularity"] < MIN_CIRCULARITY or cell["eccentricity"] > MAX_ECCENTRICITY:
                continue
            cell["local_contrast"] = local_contrast(coords, filled)
            if cell["local_contrast"] < MIN_CONTRAST:  # לא בולט מעל הרקמה שסביבו
                continue
            masks[coords[:, 0], coords[:, 1]] = idx
            info[idx] = cell
            idx += 1

        return masks, info, idx

    def detect_at_click(cy, cx):
        """
        זיהוי תא סביב נקודת הקליק: חיפוש בעיגול (לא בריבוע), העלאת סף הדרגתית
        עד שהתא נפרד מהרקמה שסביבו. מחזיר (קואורדינטות, info) או None.
        """
        r = manual_search_radius_px
        y0, y1 = max(0, cy - r), min(h_dim, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w_dim, cx + r + 1)
        crop = thresh_img[y0:y1, x0:x1]
        ry, rx = cy - y0, cx - x0
        yy, xx = np.ogrid[:crop.shape[0], :crop.shape[1]]
        d2 = (yy - ry) ** 2 + (xx - rx) ** 2
        search_disk = d2 <= r ** 2
        disk_edge = search_disk & (d2 >= (r - 2) ** 2)

        click_val = float(np.median(crop[max(0, ry - 2):ry + 3, max(0, rx - 2):rx + 3]))
        t_lo = max(0.02, current_thresh[0] * 0.70)
        t_hi = max(t_lo + 0.01, click_val * 0.95)
        for t in np.linspace(t_lo, t_hi, 10):
            b = (crop >= t) & search_disk
            b = binary_fill_holes(opening(closing(b, disk(2)), disk(1)))
            lab = split_touching_cells(b, norm_mip[y0:y1, x0:x1], pixel_size_um)
            picked = lab[ry, rx]
            if picked == 0:
                continue
            piece = lab == picked
            if np.any(piece & disk_edge):  # דולף לרקמה מסביב – להעלות סף
                continue
            if not (0.75 * MIN_DIAM_UM <= diam_of(piece.sum()) <= MAX_DIAM_UM):
                continue
            return describe(regionprops(piece.astype(np.uint8))[0], y0, x0, is_manual=True)
        return None

    def find_hit(cy, cx):
        hit_id = candidate_masks[cy, cx]
        if hit_id == 0:
            sub = candidate_masks[max(0, cy - 12):min(h_dim, cy + 13), max(0, cx - 12):min(w_dim, cx + 13)]
            valid = sub[sub > 0]
            if len(valid) > 0:
                counts = np.bincount(valid.flatten())
                counts[0] = 0
                hit_id = counts.argmax()
        return int(hit_id)

    def add_cell(coords, cell):
        new_id = next_manual_id[0]
        next_manual_id[0] += 1
        free = candidate_masks[coords[:, 0], coords[:, 1]] == 0
        candidate_masks[coords[free, 0], coords[free, 1]] = new_id
        candidate_info[new_id] = cell
        active_ids.add(new_id)

    current_thresh = [default_thresh]
    candidate_masks, candidate_info, next_id = segment_at_threshold(current_thresh[0])
    active_ids = set(candidate_info.keys())
    next_manual_id = [next_id]

    fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    plt.subplots_adjust(bottom=0.18)
    ax[0].axis("off")
    ax[1].axis("off")

    def colorize(intensity):
        rgb = intensity[:, :, None] * base_col[None, None, :]
        return (np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)

    def get_view():
        display_intensity = norm_mip.copy()
        display_intensity[thresh_img < current_thresh[0]] *= 0.15
        overlay = colorize(display_intensity)
        overlay[cell_outlines(active_ids)] = (HIGHLIGHT_COLOR * 255).astype(np.uint8)
        return overlay, colorize(norm_mip)

    def cell_outlines(ids):
        """קו מתאר לכל תא בנפרד – כולל הקו שבין שני תאים צמודים."""
        if not ids:
            return np.zeros((h_dim, w_dim), dtype=bool)
        shown = np.where(np.isin(candidate_masks, np.array(list(ids))), candidate_masks, 0)
        return find_boundaries(shown, mode="inner")

    img_raw = ax[0].imshow(np.zeros((h_dim, w_dim, 3), dtype=np.uint8))
    img_eval = ax[1].imshow(np.zeros((h_dim, w_dim, 3), dtype=np.uint8))
    text_objects = []

    def refresh():
        over_img, raw_img = get_view()
        img_raw.set_data(raw_img)
        ax[0].set_title(f"[{file_path.parent.name}/{file_path.name}] Ch {ch_idx} Raw MIP", fontweight="bold")
        img_eval.set_data(over_img)

        for t in text_objects:
            t.remove()
        text_objects.clear()
        for cid in active_ids:
            info = candidate_info[cid]
            t = ax[1].text(info["cx"] + 4, info["cy"] - 4, f"{cid}\n({info['diam_um']:.1f}µm)",
                           color="#ffffff", fontsize=7, fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.75, edgecolor="#00ffff"))
            text_objects.append(t)

        ax[1].set_title(
            f"Ch {ch_idx} ({CHANNEL_NAMES.get(ch_idx, ch_idx)}) | Thresh: {current_thresh[0]:.2f} | "
            f"{len(active_ids)} cells\n"
            "[Slider: Adjust | L-Click: Add/Restore | R-Click: Delete | x: Split | Close: Save & Next]",
            fontweight="bold", fontsize=11)
        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    thresh_slider = Slider(ax_slider, "Threshold", 0.02, 0.80, valinit=default_thresh, valstep=0.01)

    def on_slider(val):
        nonlocal candidate_masks, candidate_info, active_ids
        current_thresh[0] = float(val)
        candidate_masks, candidate_info, next_manual_id[0] = segment_at_threshold(val)
        active_ids = set(candidate_info.keys())
        refresh()

    thresh_slider.on_changed(on_slider)

    def on_click(event):
        if event.inaxes != ax[1] or event.xdata is None or event.ydata is None:
            return
        cx_full, cy_full = int(round(event.xdata)), int(round(event.ydata))
        hit_id = find_hit(cy_full, cx_full)

        if event.button == 1:  # הוספה מקומית חכמה או החזרה
            if hit_id == 0:
                detected = detect_at_click(cy_full, cx_full)
                if detected is None:
                    # לא נמצא גבול ברור – עיגול בגודל תא בינוני (~28um)
                    print(f"  (no clear cell edge at x={cx_full}, y={cy_full} - added a 28 µm circle)")
                    fallback_r = int(round((28.0 / 2.0) / pixel_size_um))
                    y0, x0 = max(0, cy_full - fallback_r), max(0, cx_full - fallback_r)
                    yy, xx = np.ogrid[y0:min(h_dim, cy_full + fallback_r + 1), x0:min(w_dim, cx_full + fallback_r + 1)]
                    circ = ((xx - cx_full) ** 2 + (yy - cy_full) ** 2) <= fallback_r ** 2
                    detected = describe(regionprops(circ.astype(np.uint8))[0], y0, x0, is_manual=True)
                add_cell(*detected)
                refresh()
            elif hit_id not in active_ids:
                active_ids.add(hit_id)
                refresh()

        elif event.button == 3:  # מחיקה
            if hit_id > 0 and hit_id in active_ids:
                active_ids.discard(hit_id)
                refresh()

    def on_key(event):
        """x = פיצול ידני של התא שמתחת לעכבר (כשכמה תאים זוהו כתא אחד)."""
        if event.key != "x" or event.inaxes != ax[1] or event.xdata is None:
            return
        hit_id = find_hit(int(round(event.ydata)), int(round(event.xdata)))
        if hit_id == 0:
            return
        ys, xs = np.nonzero(candidate_masks == hit_id)
        y0, x0 = ys.min(), xs.min()
        piece = np.zeros((ys.max() - y0 + 1, xs.max() - x0 + 1), dtype=bool)
        piece[ys - y0, xs - x0] = True
        parts = force_split(piece, pixel_size_um)
        props = [p for p in regionprops(parts) if p.area >= MIN_SIZE_PX]
        if len(props) < 2:
            print(f"  (cell {hit_id}: could not find separate centres to split)")
            return
        candidate_masks[ys, xs] = 0
        candidate_info.pop(hit_id, None)
        active_ids.discard(hit_id)
        for p in props:
            add_cell(*describe(p, y0, x0, is_manual=True))
        refresh()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh()
    plt.show()

    final_ids = sorted(active_ids)

    # תמונת תוצאה לתיעוד
    result_img = colorize(norm_mip)
    result_img[cell_outlines(final_ids)] = (HIGHLIGHT_COLOR * 255).astype(np.uint8)
    for cid in final_ids:
        info = candidate_info[cid]
        cv2.putText(result_img, f"{info['diam_um']:.0f}", (int(info["cx"]) + 4, int(info["cy"]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result_img, f"{file_path.stem} Ch{ch_idx}: {len(final_ids)} cells (MIP)",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    prefix = f"{file_path.parent.parent.name}_{file_path.parent.name}_{file_path.stem}"
    result_path = RUN_OUTPUT_DIR / f"{prefix}_ch{ch_idx}_MIP_RESULT.png"
    cv2.imwrite(str(result_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"  Ch {ch_idx}: {len(final_ids)} cells -> {result_path}")

    per_cell_rows = [{
        "cell_id": cid,
        "diam_um": round(candidate_info[cid]["diam_um"], 2),
        "major_diam_um": round(candidate_info[cid]["major_diam_um"], 2),
        "minor_diam_um": round(candidate_info[cid]["minor_diam_um"], 2),
        "area_um2": round(candidate_info[cid]["area_um2"], 2),
        "circularity": round(candidate_info[cid]["circularity"], 2),
        "eccentricity": round(candidate_info[cid]["eccentricity"], 2),
        "brightness": round(candidate_info[cid]["brightness"], 4),
        "local_contrast": round(candidate_info[cid].get("local_contrast", np.nan), 3),
        "is_manual": candidate_info[cid]["is_manual"],
        "centroid_y_px": round(candidate_info[cid]["cy"], 1),
        "centroid_x_px": round(candidate_info[cid]["cx"], 1),
    } for cid in final_ids]

    return len(final_ids), per_cell_rows


def process_file(file_path):
    print(f"\n{'#' * 55}\n# FILE: {file_path}\n{'#' * 55}")

    with nd2.ND2File(str(file_path)) as f:
        vs = f.voxel_size() if hasattr(f, "voxel_size") else None
        vx = vs.x if vs is not None and vs.x else None
        vy = vs.y if vs is not None and vs.y else None
        raw_stack = f.asarray()
        c_axis = list(f.sizes.keys()).index("C") if "C" in f.sizes else None
        channel_colors = {ch: get_nd2_channel_color(f, ch) for ch in CHANNELS}

    if vx is None or vy is None:
        print("  !! WARNING: could not read voxel size from ND2 metadata - falling back to 0.5 um/px.")
        vx, vy = vx or 0.5, vy or 0.5

    pixel_size_um = float(vx)
    pixel_area_um2 = vx * vy
    temp_ch = np.take(raw_stack, indices=CHANNELS[0], axis=c_axis) if c_axis is not None else raw_stack
    _, h_dim, w_dim = temp_ch.shape

    file_meta = {
        "experiment": file_path.parent.parent.name,
        "parent_folder": file_path.parent.name,
        "file": file_path.name,
        "file_path": str(file_path.resolve()),
    }

    count_rows, cell_rows = [], []
    for ch in CHANNELS:
        cnt, rows = process_channel_mip(
            file_path, ch, raw_stack, c_axis, pixel_area_um2, pixel_size_um,
            h_dim, w_dim, channel_colors.get(ch, np.array([1.0, 1.0, 1.0])),
        )
        count_rows.append({**file_meta, "channel": ch, "cell_count": cnt})
        cell_rows.extend({**file_meta, "channel": ch, **r} for r in rows)

    return count_rows, cell_rows


# ==========================================
# SUMMARY TABLE
# ==========================================
def add_derived_columns(df):
    df = df.copy()
    df["drg_level"] = df["file"].apply(extract_drg_level)
    df["channel_name"] = df["channel"].map(CHANNEL_NAMES).fillna(df["channel"].astype(str))
    if "diam_um" in df.columns:
        df["fiber_type"] = df["diam_um"].apply(classify_soma_size)
    return df


def build_level_counts(counts_df, levels, ch_names):
    """טבלת ספירה: שורה לכל רמת DRG, עמודה לכל ערוץ (+ מספר דגימות)."""
    table = (counts_df.pivot_table(index="drg_level", columns="channel_name", values="cell_count", aggfunc="sum")
             .reindex(index=levels, columns=ch_names).fillna(0).astype(int))
    table["Total"] = table.sum(axis=1)
    table["n_samples"] = counts_df.groupby("drg_level")["file_path"].nunique().reindex(levels).fillna(0).astype(int)
    table.loc["ALL"] = table.sum()
    return table


def build_channel_diameter_summary(cells_df, ch_names):
    """טבלת קוטר: שורה לכל ערוץ – סטטיסטיקת קוטר וחלוקה לסוגי סיבים."""
    stats = cells_df.groupby("channel_name")["diam_um"].agg(
        n_cells="count", mean_diam_um="mean", median_diam_um="median", sd_diam_um="std",
        min_diam_um="min", max_diam_um="max")
    fibers = (cells_df.groupby(["channel_name", "fiber_type"]).size().unstack(fill_value=0)
              .reindex(columns=FIBER_ORDER, fill_value=0))
    summary = pd.concat([stats, fibers], axis=1).reindex(ch_names)
    summary[FIBER_ORDER] = summary[FIBER_ORDER].fillna(0).astype(int)
    summary["n_cells"] = summary["n_cells"].fillna(0).astype(int)
    for fiber in FIBER_ORDER:
        summary[f"{fiber}_%"] = np.where(summary["n_cells"] > 0,
                                         100.0 * summary[fiber] / summary["n_cells"].clip(lower=1), 0.0)
    return summary.round(2)


# ==========================================
# PLOTS
# ==========================================
def set_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(fig, name):
    fig.savefig(RUN_OUTPUT_DIR / f"{name}.png", dpi=300)
    fig.savefig(RUN_OUTPUT_DIR / f"{name}.pdf", format="pdf")


def plot_counts_per_level(counts_df, levels, ch_names, title_suffix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # שמאל: סה"כ תאים לכל רמה ולכל ערוץ
    totals = (counts_df.pivot_table(index="drg_level", columns="channel_name", values="cell_count", aggfunc="sum")
              .reindex(index=levels, columns=ch_names).fillna(0).astype(int))
    x = np.arange(len(levels))
    w = 0.8 / len(ch_names)
    for i, ch in enumerate(ch_names):
        bars = ax1.bar(x + (i - (len(ch_names) - 1) / 2) * w, totals[ch], width=w,
                       color=CHANNEL_COLORS.get(ch), edgecolor="black", linewidth=0.7, label=ch)
        ax1.bar_label(bars, padding=2, fontsize=8.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels, fontweight="bold")
    ax1.set_xlabel("DRG Level", fontweight="bold")
    ax1.set_ylabel("Total Cell Count", fontweight="bold")
    ax1.set_title(f"Total Traced Somata per DRG Level{title_suffix}", fontweight="bold")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(frameon=False)

    # ימין: ממוצע ± SEM על פני הדגימות (כל קובץ = נקודה)
    sns.barplot(data=counts_df, x="drg_level", y="cell_count", hue="channel_name",
                order=levels, hue_order=ch_names, palette=CHANNEL_COLORS,
                errorbar="se", capsize=0.15, err_kws={"linewidth": 1.2, "color": "black"},
                alpha=0.45, edgecolor="black", linewidth=0.8, ax=ax2)
    sns.stripplot(data=counts_df, x="drg_level", y="cell_count", hue="channel_name",
                  order=levels, hue_order=ch_names, dodge=True, jitter=0.08, size=5.5,
                  palette=CHANNEL_COLORS, edgecolor="black", linewidth=0.8, legend=False, ax=ax2)
    ax2.set_xlabel("DRG Level", fontweight="bold")
    ax2.set_ylabel("Cells per Sample (mean ± SEM)", fontweight="bold")
    ax2.set_title("Per-Sample Counts (each dot = one .nd2 file)", fontweight="bold")
    ax2.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=False, title=None)

    fig.tight_layout()
    save_fig(fig, "counts_per_drg_level")


def plot_diameter_distribution(cells_df, ch_names, title_suffix):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})

    # שמאל: היסטוגרמה + KDE לכל ערוץ
    bin_width = 2.5
    lo = np.floor(min(MIN_DIAM_UM, cells_df["diam_um"].min()) / bin_width) * bin_width
    hi = np.ceil(max(MAX_DIAM_UM, cells_df["diam_um"].max()) / bin_width) * bin_width
    bins = np.arange(lo, hi + bin_width, bin_width)
    centers = bins[:-1] + bin_width / 2
    x_grid = np.linspace(lo, hi, 300)
    w = bin_width / len(ch_names)

    for i, ch in enumerate(ch_names):
        diams = cells_df.loc[cells_df["channel_name"] == ch, "diam_um"].dropna().values
        if len(diams) == 0:
            continue
        color = CHANNEL_COLORS.get(ch)
        hist, _ = np.histogram(diams, bins=bins)
        ax1.bar(centers - bin_width / 2 + (i + 0.5) * w, hist, width=w * 0.9, color=color, alpha=0.45,
                edgecolor=color, label=f"{ch} (n={len(diams)}, median {np.median(diams):.1f} µm)")
        if len(diams) >= 3 and np.std(diams) > 0:
            kde = gaussian_kde(diams, bw_method=0.35)
            ax1.plot(x_grid, kde(x_grid) * len(diams) * bin_width, color=color, lw=2.2)

    for thr in (C_FIBER_MAX_DIAM, A_DELTA_MAX_DIAM):
        ax1.axvline(thr, color="#888888", linestyle="--", lw=1.1)
    y_top = ax1.get_ylim()[1] * 1.25
    ax1.set_ylim(0, y_top)
    box = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#cccccc", alpha=0.85)
    for x_txt, txt in [((lo + C_FIBER_MAX_DIAM) / 2, f"C-fibers\n(≤{C_FIBER_MAX_DIAM:g} µm)"),
                       ((C_FIBER_MAX_DIAM + A_DELTA_MAX_DIAM) / 2,
                        f"Aδ-fibers\n({C_FIBER_MAX_DIAM:g}–{A_DELTA_MAX_DIAM:g} µm)"),
                       ((A_DELTA_MAX_DIAM + hi) / 2, f"Aβ-fibers\n(>{A_DELTA_MAX_DIAM:g} µm)")]:
        ax1.text(x_txt, y_top * 0.97, txt, ha="center", va="top", fontsize=8.5, fontweight="bold",
                 color="#333333", bbox=box)
    ax1.set_xlim(lo, hi)
    ax1.set_xlabel("Soma Diameter (µm)", fontweight="bold")
    ax1.set_ylabel("Number of Somata", fontweight="bold")
    ax1.set_title(f"Soma Diameter Distribution by Injection Target{title_suffix}", fontweight="bold")
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#cccccc", loc="center right", fontsize=8.5)

    # ימין: ספירת סוגי סיבים לכל ערוץ
    fiber_table = (cells_df.groupby(["channel_name", "fiber_type"]).size().unstack(fill_value=0)
                   .reindex(index=ch_names, columns=FIBER_ORDER, fill_value=0))
    x = np.arange(len(ch_names))
    bw = 0.25
    for i, fiber in enumerate(FIBER_ORDER):
        bars = ax2.bar(x + (i - 1) * bw * 1.05, fiber_table[fiber], width=bw, label=fiber,
                       color=FIBER_COLORS[fiber], edgecolor="black", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                             textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ch_names, fontweight="semibold")
    ax2.set_ylabel("Somata Count", fontweight="bold")
    ax2.set_title("Fiber Type by Channel", fontweight="bold")
    ax2.set_ylim(0, max(1, fiber_table.values.max()) * 1.2)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    save_fig(fig, "diameter_distribution_and_fiber_types")


def summarize_and_plot(side_filter=None):
    if not MASTER_COUNTS_CSV.exists():
        print(f"No master counts found at {MASTER_COUNTS_CSV} - process some .nd2 files first.")
        return

    counts_df = add_derived_columns(pd.read_csv(MASTER_COUNTS_CSV))
    if MASTER_PER_CELL_CSV.exists() and MASTER_PER_CELL_CSV.stat().st_size > 0:
        cells_df = add_derived_columns(pd.read_csv(MASTER_PER_CELL_CSV))
    else:
        cells_df = add_derived_columns(pd.DataFrame(columns=["file", "file_path", "channel", "parent_folder",
                                                             "diam_um"]))

    title_suffix = ""
    if side_filter:
        counts_df = counts_df[counts_df["parent_folder"] == side_filter]
        cells_df = cells_df[cells_df["parent_folder"] == side_filter]
        title_suffix = f" ({side_filter})"
    if counts_df.empty:
        print("No data left after filtering.")
        return

    levels = sorted(counts_df["drg_level"].unique(), key=drg_sort_key)
    ch_names = [CHANNEL_NAMES.get(ch, str(ch)) for ch in CHANNELS]

    level_counts = build_level_counts(counts_df, levels, ch_names)
    counts_path = RUN_OUTPUT_DIR / "counts_per_drg_level.csv"
    level_counts.to_csv(counts_path, index_label="drg_level", encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print(f"CELL COUNT PER DRG LEVEL{title_suffix}")
    print("=" * 70)
    print(level_counts.to_string())
    print(f"Saved: {counts_path}")

    if not cells_df.empty:
        diam_summary = build_channel_diameter_summary(cells_df, ch_names)
        diam_path = RUN_OUTPUT_DIR / "diameter_per_channel.csv"
        diam_summary.to_csv(diam_path, index_label="channel_name", encoding="utf-8-sig")
        print("\n" + "=" * 70)
        print(f"SOMA DIAMETER PER CHANNEL{title_suffix}")
        print("=" * 70)
        print(diam_summary.to_string())
        print(f"Saved: {diam_path}")

    set_plot_style()
    plot_counts_per_level(counts_df, levels, ch_names, title_suffix)
    if not cells_df.empty:
        plot_diameter_distribution(cells_df, ch_names, title_suffix)
    else:
        print("No per-cell diameter data available - skipping diameter plots.")
    print(f"Saved plots to: {RUN_OUTPUT_DIR}")
    plt.show()


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nWhat do you want to do?")
    action = input("  [p]rocess .nd2 files then plot / [s]ummary & plots only (default: p): ").strip().lower()

    if action not in ("s", "summary"):
        mode = input("  [f]ile / [d]irectory (default: directory): ").strip().lower()
        if mode in ("f", "file"):
            typed = input("  Path to the .nd2 file: ").strip()
        else:
            typed = input(f"  Path to the folder [{DEFAULT_INPUT_PATH}]: ").strip()
        input_path = Path(typed) if typed else Path(DEFAULT_INPUT_PATH)

        if not input_path.exists():
            print(f"Error: {input_path} not found.")
        else:
            nd2_files = sorted(input_path.rglob("*.nd2")) if input_path.is_dir() else [input_path]
            if not nd2_files:
                print(f"No .nd2 files found under: {input_path}")

            run_counts, run_cells = [], []
            for fp in nd2_files:
                try:
                    counts, cells = process_file(fp)
                except Exception as e:
                    print(f"  !! Error processing {fp.name}: {e}")
                    continue
                run_counts.extend(counts)
                run_cells.extend(cells)

            if run_counts:
                run_counts_df = pd.DataFrame(run_counts)
                run_cells_df = pd.DataFrame(run_cells)
                run_counts_df.to_csv(RUN_OUTPUT_DIR / "run_counts.csv", index=False, encoding="utf-8-sig")
                run_cells_df.to_csv(RUN_OUTPUT_DIR / "run_per_cell.csv", index=False, encoding="utf-8-sig")

                sort_cols = ["experiment", "parent_folder", "file", "channel"]
                upsert_master(run_counts_df, MASTER_COUNTS_CSV, sort_cols)
                if not run_cells_df.empty:
                    upsert_master(run_cells_df, MASTER_PER_CELL_CSV, sort_cols + ["cell_id"])
                else:
                    # כל הערוצים יצאו 0 תאים – עדיין צריך למחוק תאים ישנים של אותם קבצים מהמאסטר
                    if MASTER_PER_CELL_CSV.exists():
                        old = pd.read_csv(MASTER_PER_CELL_CSV)
                        keys = set(zip(run_counts_df["file_path"], run_counts_df["channel"]))
                        old = old[[(fp, ch) not in keys for fp, ch in zip(old["file_path"], old["channel"])]]
                        old.to_csv(MASTER_PER_CELL_CSV, index=False, encoding="utf-8-sig")

                print("\n" + "=" * 60)
                print("CURRENT RUN COUNTS:")
                print(run_counts_df[["experiment", "parent_folder", "file", "channel", "cell_count"]]
                      .to_string(index=False))
                print(f"Masters updated: {MASTER_COUNTS_CSV}, {MASTER_PER_CELL_CSV}")

    side = input("\nPlot which side? [Lt / Rt / blank = all]: ").strip()
    summarize_and_plot(side_filter=side or None)
