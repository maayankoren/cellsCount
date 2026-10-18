# """
# DRG Tracer Cell Counter & Soma Sizing - Full Cytoplasm MIP Version
# Features:
#   - Treats WGA signal as filling the whole soma cytoplasm.
#   - Automatically bridges intra-cellular gaps and fills nuclear cavities.
#   - Computes equivalent diameter, major/minor ellipse axes, and area in um.
#   - Smart local segmentation on manual clicks (no arbitrary fixed-radius circles).
#   - Live visual feedback of measured cell diameter (in um) on the overlay.
#   - Non-destructive append mode to master CSVs and per-run versioning.
# """

# from datetime import datetime
# from pathlib import Path
# import cv2
# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider
# import nd2
# import numpy as np
# import pandas as pd
# from scipy.ndimage import binary_fill_holes
# from skimage.measure import label, regionprops
# from skimage.morphology import closing, disk, dilation, opening
# from skimage.segmentation import find_boundaries

# # ==========================================
# # CONFIG
# # ==========================================
# INPUT_PATH = "drg_samples"
# CHANNELS = [1, 2]

# # גבולות סינון קוטר תא במיקרונים
# MIN_DIAM_UM = 16.0
# MAX_DIAM_UM = 55.0
# BORDER_MARGIN_PX = 20

# # פילטר צורת תא (מאפשר תאים עגולים/אליפטיים וחוסם מריחות רקמה וסיבים)
# MIN_CIRCULARITY = 0.22
# MAX_ECCENTRICITY = 0.88

# CLOSING_RADIUS_PX = 3
# OPENING_RADIUS_PX = 1
# MIN_SIZE_PX = 80

# SAVE_DEBUG_OVERLAY = True
# HIGHLIGHT_COLOR = np.array([0.0, 1.0, 1.0])
# DEBUG_COLORS = {
#     "passed": np.array([0.0, 1.0, 0.0]),
#     "failed_border": np.array([1.0, 0.0, 0.0]),
#     "failed_size": np.array([1.0, 0.6, 0.0]),
#     "failed_shape": np.array([1.0, 0.0, 1.0]),
# }

# BASE_OUTPUT_DIR = Path("drg_mip_outputs")
# BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# OUTPUT_DIR = BASE_OUTPUT_DIR / f"run_{timestamp}"
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# print(f"Outputs will be saved to: {OUTPUT_DIR}")


# def append_to_csv(df, target_path):
#     """שומר שורות לקובץ CSV מבלי לדרוס נתונים קודמים."""
#     target_path = Path(target_path)
#     target_path.parent.mkdir(parents=True, exist_ok=True)
#     write_header = not (target_path.exists() and target_path.stat().st_size > 0)
#     df.to_csv(target_path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


# def get_nd2_channel_color(f_nd2, ch_idx):
#     """חילוץ צבע הערוץ המוגדר מתוך המטא-דאטה של קובץ ה-ND2."""
#     try:
#         channels_meta = f_nd2.metadata.channels
#         if channels_meta and ch_idx < len(channels_meta):
#             ch = channels_meta[ch_idx].channel
#             c_obj = getattr(ch, "colorRGB", None) or getattr(ch, "color", None)

#             if c_obj is not None:
#                 if hasattr(c_obj, "r") and hasattr(c_obj, "g") and hasattr(c_obj, "b"):
#                     r, g, b = c_obj.r, c_obj.g, c_obj.b
#                     if max(r, g, b) > 1.0:
#                         r, g, b = r / 255.0, g / 255.0, b / 255.0
#                     return np.array([r, g, b], dtype=np.float32)

#                 if isinstance(c_obj, (list, tuple)) and len(c_obj) >= 3:
#                     r, g, b = c_obj[0], c_obj[1], c_obj[2]
#                     if max(r, g, b) > 1.0:
#                         r, g, b = r / 255.0, g / 255.0, b / 255.0
#                     return np.array([r, g, b], dtype=np.float32)

#                 color_int = int(c_obj) & 0xFFFFFF
#                 r = ((color_int >> 16) & 0xFF) / 255.0
#                 g = ((color_int >> 8) & 0xFF) / 255.0
#                 b = (color_int & 0xFF) / 255.0
#                 if r == 0 and g == 0 and b == 0:
#                     return np.array([1.0, 1.0, 1.0])
#                 return np.array([r, g, b], dtype=np.float32)
#     except Exception as e:
#         print(f"Error reading channel {ch_idx} color: {e}")

#     fallback_map = {0: [0.2, 0.5, 1.0], 1: [1.0, 0.1, 0.1], 2: [0.1, 1.0, 0.1]}
#     return np.array(fallback_map.get(ch_idx, [1.0, 1.0, 1.0]), dtype=np.float32)


# def calculate_default_threshold(norm_mip):
#     """חישוב סף דינמי ראשוני על סמך רמת רעש הרקע."""
#     tissue_fraction = np.mean(norm_mip > 0.10)
#     if tissue_fraction < 0.15:
#         return 0.12

#     tissue_pixels = norm_mip[norm_mip > 0.05]
#     if len(tissue_pixels) == 0:
#         return 0.25
#     computed = float(np.median(tissue_pixels) + 0.55 * np.std(tissue_pixels))
#     return float(np.clip(computed, 0.22, 0.55))


# def process_channel_mip(file_path, ch_idx, raw_stack, c_axis, pixel_area_um2, pixel_size_um, h_dim, w_dim, base_col):
#     print(f"\n{'=' * 55}")
#     print(f">>> Tracer Channel {ch_idx}: MIP dynamic segmentation & soma sizing")
#     print(f"    Color assigned from metadata: RGB = {np.round(base_col, 2)}")
#     print(f"    Pixel calibration: {pixel_size_um:.4f} um/px")
#     print("=" * 55)

#     prefix = f"{file_path.parent.name}_{file_path.stem}"

#     raw_tr = np.take(raw_stack, indices=ch_idx, axis=c_axis) if c_axis is not None else raw_stack
#     mip_raw = raw_tr.max(axis=0).astype(np.float32)

#     p_l, p_h = np.percentile(mip_raw, (2.0, 99.8))
#     norm_mip = np.clip((mip_raw - p_l) / (p_h - p_l + 1e-5), 0, 1)

#     default_thresh = calculate_default_threshold(norm_mip)
#     print(f"  Dynamic default threshold: {default_thresh:.2f}")

#     manual_search_radius_px = int(round((MAX_DIAM_UM / 2.0) / pixel_size_um))

#     def segment_at_threshold(thresh_val):
#         binary = norm_mip >= thresh_val
#         if CLOSING_RADIUS_PX > 0:
#             binary = closing(binary, disk(CLOSING_RADIUS_PX))
#         if OPENING_RADIUS_PX > 0:
#             binary = opening(binary, disk(OPENING_RADIUS_PX))

#         m = label(binary, connectivity=2)
#         masks = np.zeros_like(m, dtype=np.int32)
#         info = {}
#         idx = 1
#         debug_list = []

#         for prop in regionprops(m, intensity_image=norm_mip):
#             if prop.area < MIN_SIZE_PX:
#                 continue

#             cy, cx = prop.centroid
#             mask_bool = (m == prop.label)

#             # מילוי חלל הציטופלזמה והגרעין לקבלת שטח הסומה המלא
#             filled_mask = binary_fill_holes(mask_bool)
#             filled_area = int(np.sum(filled_mask))

#             # חישוב מדדי קוטר
#             diam_um = 2.0 * np.sqrt((filled_area * pixel_area_um2) / np.pi)
#             major_diam_um = prop.major_axis_length * pixel_size_um
#             minor_diam_um = prop.minor_axis_length * pixel_size_um

#             perim = prop.perimeter
#             circularity = (4.0 * np.pi * filled_area) / (perim ** 2 + 1e-6)
#             eccentricity = prop.eccentricity

#             row = {
#                 "raw_label": int(prop.label), "cy": round(cy, 1), "cx": round(cx, 1),
#                 "diam_um": round(diam_um, 2),
#                 "major_diam_um": round(major_diam_um, 2),
#                 "minor_diam_um": round(minor_diam_um, 2),
#                 "circularity": round(circularity, 2),
#                 "eccentricity": round(eccentricity, 2), "reason": None,
#             }

#             if (cy < BORDER_MARGIN_PX or cy > h_dim - BORDER_MARGIN_PX or
#                     cx < BORDER_MARGIN_PX or cx > w_dim - BORDER_MARGIN_PX):
#                 row["reason"] = "failed_border"
#                 debug_list.append(row)
#                 continue

#             if not (MIN_DIAM_UM <= diam_um <= MAX_DIAM_UM):
#                 row["reason"] = "failed_size"
#                 debug_list.append(row)
#                 continue

#             if circularity < MIN_CIRCULARITY or eccentricity > MAX_ECCENTRICITY:
#                 row["reason"] = "failed_shape"
#                 debug_list.append(row)
#                 continue

#             row["reason"] = "passed"
#             debug_list.append(row)

#             masks[filled_mask] = idx
#             info[idx] = {
#                 "cy": cy, "cx": cx,
#                 "diam_um": diam_um,
#                 "major_diam_um": major_diam_um,
#                 "minor_diam_um": minor_diam_um,
#                 "area_um2": filled_area * pixel_area_um2,
#                 "brightness": float(np.percentile(norm_mip[mask_bool], 90)),
#                 "circularity": circularity,
#                 "eccentricity": eccentricity,
#                 "is_manual": False,
#                 "raw_label": int(prop.label),
#             }
#             idx += 1

#         return masks, info, idx, debug_list, m

#     current_thresh = [default_thresh]
#     candidate_masks, candidate_info, next_id, debug_rows, m_label = segment_at_threshold(current_thresh[0])
#     active_ids = set(candidate_info.keys())
#     next_manual_id = [next_id]

#     fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
#     plt.subplots_adjust(bottom=0.18)
#     ax[0].axis("off")
#     ax[1].axis("off")

#     def get_view():
#         display_intensity = norm_mip.copy()
#         bg_mask = norm_mip < current_thresh[0]
#         display_intensity[bg_mask] *= 0.15

#         rgb = np.zeros((h_dim, w_dim, 3), dtype=np.float32)
#         rgb[:, :, 0] = display_intensity * base_col[0]
#         rgb[:, :, 1] = display_intensity * base_col[1]
#         rgb[:, :, 2] = display_intensity * base_col[2]
#         overlay = (np.clip(rgb, 0, 1) * 255.0).copy()

#         if active_ids:
#             valid_mask = np.isin(candidate_masks, np.array(list(active_ids)))
#             bounds = find_boundaries(valid_mask, mode="inner")
#             bounds = dilation(bounds)
#             overlay[bounds] = HIGHLIGHT_COLOR * 255.0

#         raw_rgb = np.zeros((h_dim, w_dim, 3), dtype=np.float32)
#         raw_rgb[:, :, 0] = norm_mip * base_col[0]
#         raw_rgb[:, :, 1] = norm_mip * base_col[1]
#         raw_rgb[:, :, 2] = norm_mip * base_col[2]

#         return np.clip(overlay, 0, 255).astype(np.uint8), (np.clip(raw_rgb, 0, 1) * 255.0).astype(np.uint8)

#     img_raw = ax[0].imshow(np.zeros((h_dim, w_dim, 3), dtype=np.uint8))
#     img_eval = ax[1].imshow(np.zeros((h_dim, w_dim, 3), dtype=np.uint8))
#     text_objects = []

#     def refresh():
#         over_img, raw_img = get_view()
#         img_raw.set_data(raw_img)
#         ax[0].set_title(f"[{file_path.name}] Ch {ch_idx} Raw MIP", fontweight="bold")
#         img_eval.set_data(over_img)

#         for t in text_objects:
#             t.remove()
#         text_objects.clear()

#         for cid in active_ids:
#             info = candidate_info[cid]
#             label_text = f"{cid}\n({info['diam_um']:.1f}µm)"
#             t = ax[1].text(info["cx"] + 4, info["cy"] - 4, label_text, color="#ffffff", fontsize=7,
#                            fontweight="bold",
#                            bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.75, edgecolor="#00ffff"))
#             text_objects.append(t)

#         ax[1].set_title(
#             f"Ch {ch_idx} MIP | Thresh: {current_thresh[0]:.2f} | {len(active_ids)} cells\n"
#             "[Slider: Adjust | L-Click: Add/Restore | R-Click: Delete | Close: Save & Next]",
#             fontweight="bold", fontsize=11)
#         fig.canvas.draw_idle()

#     ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
#     thresh_slider = Slider(ax_slider, "Threshold", 0.02, 0.80, valinit=default_thresh, valstep=0.01)

#     def on_slider(val):
#         nonlocal candidate_masks, candidate_info, active_ids, debug_rows, m_label
#         current_thresh[0] = float(val)
#         candidate_masks, candidate_info, next_manual_id[0], debug_rows, m_label = segment_at_threshold(val)
#         active_ids = set(candidate_info.keys())
#         refresh()

#     thresh_slider.on_changed(on_slider)

#     def on_click(event):
#         if event.inaxes != ax[1] or event.xdata is None or event.ydata is None:
#             return
#         cx_full, cy_full = int(round(event.xdata)), int(round(event.ydata))

#         hit_id = candidate_masks[cy_full, cx_full]
#         if hit_id == 0:
#             sub = candidate_masks[max(0, cy_full - 12):min(h_dim, cy_full + 13),
#                                   max(0, cx_full - 12):min(w_dim, cx_full + 13)]
#             valid = sub[sub > 0]
#             if len(valid) > 0:
#                 counts = np.bincount(valid.flatten())
#                 counts[0] = 0
#                 hit_id = counts.argmax()

#         if event.button == 1:  # הוספה מקומית חכמה או החזרה
#             if hit_id == 0:
#                 y_min = max(0, cy_full - manual_search_radius_px)
#                 y_max = min(h_dim, cy_full + manual_search_radius_px + 1)
#                 x_min = max(0, cx_full - manual_search_radius_px)
#                 x_max = min(w_dim, cx_full + manual_search_radius_px + 1)

#                 local_crop = norm_mip[y_min:y_max, x_min:x_max]
#                 local_thresh = max(0.06, current_thresh[0] * 0.70)
#                 local_bin = local_crop >= local_thresh
#                 local_bin = binary_fill_holes(closing(local_bin, disk(2)))
#                 local_labeled = label(local_bin)

#                 rel_y, rel_x = cy_full - y_min, cx_full - x_min
#                 picked_label = (
#                     local_labeled[rel_y, rel_x]
#                     if (0 <= rel_y < local_bin.shape[0] and 0 <= rel_x < local_bin.shape[1])
#                     else 0
#                 )

#                 if picked_label > 0:
#                     sub_mask = local_labeled == picked_label
#                     cell_mask = np.zeros((h_dim, w_dim), dtype=bool)
#                     cell_mask[y_min:y_max, x_min:x_max] = sub_mask

#                     props = regionprops(sub_mask.astype(int), intensity_image=local_crop)[0]
#                     area_px = int(np.sum(sub_mask))
#                     diam_um = 2.0 * np.sqrt((area_px * pixel_area_um2) / np.pi)
#                     major_diam_um = props.major_axis_length * pixel_size_um
#                     minor_diam_um = props.minor_axis_length * pixel_size_um
#                     ecc = props.eccentricity
#                 else:
#                     fallback_r = int(round((28.0 / 2.0) / pixel_size_um))  # ברירת מחדל של תא בינוני (~28um)
#                     yy, xx = np.ogrid[:h_dim, :w_dim]
#                     cell_mask = ((xx - cx_full) ** 2 + (yy - cy_full) ** 2) <= (fallback_r ** 2)
#                     area_px = int(np.sum(cell_mask))
#                     diam_um = 2.0 * np.sqrt((area_px * pixel_area_um2) / np.pi)
#                     major_diam_um = minor_diam_um = diam_um
#                     ecc = 0.0

#                 new_id = next_manual_id[0]
#                 next_manual_id[0] += 1
#                 candidate_masks[cell_mask & (candidate_masks == 0)] = new_id

#                 candidate_info[new_id] = {
#                     "cy": cy_full, "cx": cx_full,
#                     "diam_um": diam_um,
#                     "major_diam_um": major_diam_um,
#                     "minor_diam_um": minor_diam_um,
#                     "area_um2": area_px * pixel_area_um2,
#                     "brightness": float(np.percentile(norm_mip[cell_mask], 90)) if np.any(cell_mask) else 1.0,
#                     "circularity": 1.0,
#                     "eccentricity": ecc,
#                     "is_manual": True,
#                     "raw_label": -1,
#                 }
#                 active_ids.add(new_id)
#                 refresh()
#             elif hit_id not in active_ids:
#                 active_ids.add(hit_id)
#                 refresh()

#         elif event.button == 3:  # מחיקה
#             if hit_id > 0 and hit_id in active_ids:
#                 active_ids.discard(hit_id)
#                 refresh()

#     fig.canvas.mpl_connect("button_press_event", on_click)
#     refresh()
#     plt.show()

#     final_valid_ids = set(active_ids)

#     if SAVE_DEBUG_OVERLAY and debug_rows:
#         _save_debug_overlay(file_path, ch_idx, norm_mip, m_label, debug_rows, h_dim, w_dim)
#         pd.DataFrame(debug_rows).to_csv(
#             OUTPUT_DIR / f"{prefix}_ch{ch_idx}_all_candidates_debug.csv",
#             index=False, encoding="utf-8-sig",
#         )

#     result_img = np.zeros((h_dim, w_dim, 3), dtype=np.float32)
#     result_img[:, :, 0] = norm_mip * base_col[0]
#     result_img[:, :, 1] = norm_mip * base_col[1]
#     result_img[:, :, 2] = norm_mip * base_col[2]
#     result_img = np.clip(result_img * 255.0, 0, 255).astype(np.uint8)
#     if final_valid_ids:
#         valid_mask = np.isin(candidate_masks, np.array(list(final_valid_ids)))
#         bounds = find_boundaries(valid_mask, mode="inner")
#         bounds = dilation(bounds)
#         result_img[bounds] = HIGHLIGHT_COLOR * 255.0
#     cv2.putText(result_img, f"{file_path.stem} Ch{ch_idx}: {len(final_valid_ids)} cells (MIP)",
#                 (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

#     result_path = OUTPUT_DIR / f"{prefix}_ch{ch_idx}_MIP_RESULT.png"
#     cv2.imwrite(str(result_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
#     print(f"Result image saved to: {result_path} ({len(final_valid_ids)} cells)")

#     per_cell_rows = [{
#         "parent_folder": file_path.parent.name,
#         "file": file_path.name,
#         "channel": ch_idx,
#         "cell_id": cid,
#         "diam_um": round(candidate_info[cid]["diam_um"], 2),
#         "major_diam_um": round(candidate_info[cid]["major_diam_um"], 2),
#         "minor_diam_um": round(candidate_info[cid]["minor_diam_um"], 2),
#         "area_um2": round(candidate_info[cid]["area_um2"], 2),
#         "circularity": round(candidate_info[cid]["circularity"], 2),
#         "eccentricity": round(candidate_info[cid]["eccentricity"], 2),
#         "brightness": round(candidate_info[cid]["brightness"], 4),
#         "is_manual": candidate_info[cid]["is_manual"],
#         "centroid_y_px": round(candidate_info[cid]["cy"], 1),
#         "centroid_x_px": round(candidate_info[cid]["cx"], 1),
#     } for cid in final_valid_ids]

#     return len(final_valid_ids), per_cell_rows


# def _save_debug_overlay(file_path, ch_idx, norm_mip, raw_labels, debug_rows, h_dim, w_dim):
#     prefix = f"{file_path.parent.name}_{file_path.stem}"
#     base = np.stack([norm_mip, norm_mip, norm_mip], axis=-1)
#     overlay = np.clip(base * 255.0, 0, 255).astype(np.uint8).copy()

#     for row in debug_rows:
#         lbl = row["raw_label"]
#         mask_bool = raw_labels == lbl
#         if not mask_bool.any():
#             continue
#         bounds = find_boundaries(mask_bool, mode="inner")
#         bounds = dilation(bounds)
#         color = DEBUG_COLORS.get(row["reason"], np.array([1, 1, 1]))
#         overlay[bounds] = (color * 255).astype(np.uint8)
#         cv2.putText(overlay, str(lbl), (int(row["cx"]) + 4, int(row["cy"]) - 4),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

#     legend_y = 25
#     for reason, color in DEBUG_COLORS.items():
#         cv2.putText(overlay, reason, (15, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                     tuple(int(c * 255) for c in color[::-1]), 2, cv2.LINE_AA)
#         legend_y += 22

#     out_path = OUTPUT_DIR / f"{prefix}_ch{ch_idx}_DEBUG_all_candidates.png"
#     cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# def process_file(file_path):
#     print(f"\n{'#' * 55}")
#     print(f"# FILE: {file_path}")
#     print(f"{'#' * 55}")

#     prefix = f"{file_path.parent.name}_{file_path.stem}"

#     with nd2.ND2File(str(file_path)) as f:
#         vx = f.voxel_size().x if hasattr(f, "voxel_size") and f.voxel_size().x else None
#         vy = f.voxel_size().y if hasattr(f, "voxel_size") and f.voxel_size().y else None
#         raw_stack = f.asarray()
#         c_axis = list(f.sizes.keys()).index("C") if "C" in f.sizes else None

#         channel_colors = {}
#         for ch in CHANNELS:
#             channel_colors[ch] = get_nd2_channel_color(f, ch)

#     if vx is None or vy is None:
#         print("  !! WARNING: could not read voxel size from ND2 metadata - falling back to 0.5 um/px.")
#         vx = vx or 0.5
#         vy = vy or 0.5

#     pixel_size_um = float(vx)
#     pixel_area_um2 = vx * vy
#     print(f"  Using pixel size: {vx:.4f} x {vy:.4f} um/px  (pixel_area_um2 = {pixel_area_um2:.5f})")

#     temp_ch = np.take(raw_stack, indices=CHANNELS[0], axis=c_axis) if c_axis is not None else raw_stack
#     _, h_dim, w_dim = temp_ch.shape

#     all_rows = []
#     counts = {}
#     for ch in CHANNELS:
#         ch_col = channel_colors.get(ch, np.array([1.0, 1.0, 1.0]))
#         cnt, rows = process_channel_mip(
#             file_path, ch, raw_stack, c_axis,
#             pixel_area_um2, pixel_size_um,
#             h_dim, w_dim, ch_col
#         )
#         counts[ch] = cnt
#         all_rows.extend(rows)

#     print(f"\n{'=' * 55}")
#     print(f"SUMMARY - {file_path.name}")
#     for ch, cnt in counts.items():
#         print(f"  Channel {ch}: {cnt} cells")
#     print("=" * 55)

#     if all_rows:
#         df = pd.DataFrame(all_rows)
#         out_csv = OUTPUT_DIR / f"{prefix}_MIP_per_cell.csv"
#         df.to_csv(out_csv, index=False, encoding="utf-8-sig")

#     return all_rows, counts


# if __name__ == "__main__":
#     print("\nProcess a single .nd2 file, or a whole folder (recursively)?")
#     mode = input("  [f]ile / [d]irectory (default: auto-detect from INPUT_PATH): ").strip().lower()

#     if mode in ("f", "file"):
#         typed = input(f"  Path to the .nd2 file [{INPUT_PATH}]: ").strip()
#         input_path = Path(typed) if typed else Path(INPUT_PATH)
#     elif mode in ("d", "dir", "directory"):
#         typed = input(f"  Path to the folder [{INPUT_PATH}]: ").strip()
#         input_path = Path(typed) if typed else Path(INPUT_PATH)
#     else:
#         input_path = Path(INPUT_PATH)

#     if not input_path.exists():
#         print(f"Error: {input_path} not found.")
#     else:
#         nd2_files = sorted(input_path.rglob("*.nd2")) if input_path.is_dir() else [input_path]

#         if not nd2_files:
#             print(f"No .nd2 files found under {input_path}.")
#         else:
#             combined_rows = []
#             combined_counts = []

#             for fp in nd2_files:
#                 try:
#                     rows, counts = process_file(fp)
#                 except Exception as e:
#                     print(f"  !! Failed to process {fp}: {e}")
#                     continue
#                 combined_rows.extend(rows)
#                 for ch, cnt in counts.items():
#                     combined_counts.append({
#                         "parent_folder": fp.parent.name,
#                         "file": fp.name,
#                         "channel": ch,
#                         "cell_count": cnt,
#                     })

#             print(f"\n{'#' * 55}")
#             print("ALL FILES DONE")
#             print(f"{'#' * 55}")

#             MASTER_SUMMARY_CSV = BASE_OUTPUT_DIR / "ALL_FILES_summary_counts.csv"
#             MASTER_PER_CELL_CSV = BASE_OUTPUT_DIR / "ALL_FILES_per_cell.csv"

#             if combined_counts:
#                 summary_df = pd.DataFrame(combined_counts)
#                 summary_df.to_csv(OUTPUT_DIR / "ALL_FILES_summary_counts.csv", index=False, encoding="utf-8-sig")
#                 append_to_csv(summary_df, MASTER_SUMMARY_CSV)
#                 print(f"Appended counts to master: {MASTER_SUMMARY_CSV}")
#                 print(summary_df.to_string(index=False))

#             if combined_rows:
#                 combined_df = pd.DataFrame(combined_rows)
#                 combined_df.to_csv(OUTPUT_DIR / "ALL_FILES_per_cell.csv", index=False, encoding="utf-8-sig")
#                 append_to_csv(combined_df, MASTER_PER_CELL_CSV)
#                 print(f"Appended cell data to master: {MASTER_PER_CELL_CSV}")
"""
DRG Tracer Cell Counter & Soma Sizing - Full Cytoplasm MIP Version
Features:
- Treats WGA signal as filling the whole soma cytoplasm.
- Automatically bridges intra-cellular gaps and fills nuclear cavities.
- Computes equivalent diameter, major/minor ellipse axes, and area in um.
- Smart local segmentation on manual clicks (no arbitrary fixed-radius circles).
- Live visual feedback of measured cell diameter (in um) on the overlay.
- Non-destructive append mode to master CSVs and per-run versioning.
- SOMA CLASSIFICATION: C-fiber (<= 25um), A-delta (25-35um), and A-beta (> 35um).
- Publication-ready distribution & relative proportion plots.
"""

from datetime import datetime
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nd2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes
import seaborn as sns
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, dilation, opening
from skimage.segmentation import find_boundaries

# ==========================================
# CONFIG
# ==========================================
INPUT_PATH = "drg_samples"
CHANNELS = [1, 2]

# גבולות סינון קוטר תא במיקרונים
MIN_DIAM_UM = 16.0
MAX_DIAM_UM = 55.0
BORDER_MARGIN_PX = 20

# פילטר צורת תא (מאפשר תאים עגולים/אליפטיים וחוסם מריחות רקמה וסיבים)
MIN_CIRCULARITY = 0.22
MAX_ECCENTRICITY = 0.88

CLOSING_RADIUS_PX = 3
OPENING_RADIUS_PX = 1
MIN_SIZE_PX = 80

# גבולות סיווג סיבים בסומות DRG (מיקרונים)
C_FIBER_MAX_DIAM = 25.0       # <= 25 um -> C-fiber
A_DELTA_MAX_DIAM = 35.0       # 25 < diam <= 35 um -> A-delta
                              # > 35 um -> A-beta / Large A

FIBER_PALETTE = {
    "C-fiber": "#d95f02",
    "A-delta": "#7570b3",
    "A-beta/Large": "#1b9e77"
}

SAVE_DEBUG_OVERLAY = True
HIGHLIGHT_COLOR = np.array([0.0, 1.0, 1.0])
DEBUG_COLORS = {
    "passed": np.array([0.0, 1.0, 0.0]),
    "failed_border": np.array([1.0, 0.0, 0.0]),
    "failed_size": np.array([1.0, 0.6, 0.0]),
    "failed_shape": np.array([1.0, 0.0, 1.0]),
}

BASE_OUTPUT_DIR = Path("drg_mip_outputs")
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = BASE_OUTPUT_DIR / f"run_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Outputs will be saved to: {OUTPUT_DIR}")


# ==========================================
# UTILS & CLASSIFICATION
# ==========================================
def append_to_csv(df, target_path):
    """שומר שורות לקובץ CSV מבלי לדרוס נתונים קודמים."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (target_path.exists() and target_path.stat().st_size > 0)
    df.to_csv(target_path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def classify_soma_size(diam):
    """סיווג לפי קוטר שווה-ערך של סומת ה-DRG."""
    if diam <= C_FIBER_MAX_DIAM:
        return "C-fiber"
    elif diam <= A_DELTA_MAX_DIAM:
        return "A-delta"
    else:
        return "A-beta/Large"


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
                    if max(r, g, b) > 1.0:
                        r, g, b = r / 255.0, g / 255.0, b / 255.0
                    return np.array([r, g, b], dtype=np.float32)

                if isinstance(c_obj, (list, tuple)) and len(c_obj) >= 3:
                    r, g, b = c_obj[0], c_obj[1], c_obj[2]
                    if max(r, g, b) > 1.0:
                        r, g, b = r / 255.0, g / 255.0, b / 255.0
                    return np.array([r, g, b], dtype=np.float32)

                color_int = int(c_obj) & 0xFFFFFF
                r = ((color_int >> 16) & 0xFF) / 255.0
                g = ((color_int >> 8) & 0xFF) / 255.0
                b = (color_int & 0xFF) / 255.0
                if r == 0 and g == 0 and b == 0:
                    return np.array([1.0, 1.0, 1.0])
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


# ==========================================
# CORE PROCESSING
# ==========================================
def process_channel_mip(file_path, ch_idx, raw_stack, c_axis, pixel_area_um2, pixel_size_um, h_dim, w_dim, base_col):
    print(f"\n{'=' * 55}")
    print(f">>> Tracer Channel {ch_idx}: MIP dynamic segmentation & soma sizing")
    print(f"    Color assigned from metadata: RGB = {np.round(base_col, 2)}")
    print(f"    Pixel calibration: {pixel_size_um:.4f} um/px")
    print("=" * 55)

    prefix = f"{file_path.parent.name}_{file_path.stem}"

    raw_tr = np.take(raw_stack, indices=ch_idx, axis=c_axis) if c_axis is not None else raw_stack
    mip_raw = raw_tr.max(axis=0).astype(np.float32)

    p_l, p_h = np.percentile(mip_raw, (2.0, 99.8))
    norm_mip = np.clip((mip_raw - p_l) / (p_h - p_l + 1e-5), 0, 1)

    default_thresh = calculate_default_threshold(norm_mip)
    print(f"  Dynamic default threshold: {default_thresh:.2f}")

    manual_search_radius_px = int(round((MAX_DIAM_UM / 2.0) / pixel_size_um))

    def segment_at_threshold(thresh_val):
        binary = norm_mip >= thresh_val
        if CLOSING_RADIUS_PX > 0:
            binary = closing(binary, disk(CLOSING_RADIUS_PX))
        if OPENING_RADIUS_PX > 0:
            binary = opening(binary, disk(OPENING_RADIUS_PX))

        m = label(binary, connectivity=2)
        masks = np.zeros_like(m, dtype=np.int32)
        info = {}
        idx = 1
        debug_list = []

        for prop in regionprops(m, intensity_image=norm_mip):
            if prop.area < MIN_SIZE_PX:
                continue

            cy, cx = prop.centroid
            mask_bool = (m == prop.label)

            # מילוי חלל הציטופלזמה והגרעין לקבלת שטח הסומה המלא
            filled_mask = binary_fill_holes(mask_bool)
            filled_area = int(np.sum(filled_mask))

            # חישוב מדדי קוטר
            diam_um = 2.0 * np.sqrt((filled_area * pixel_area_um2) / np.pi)
            major_diam_um = prop.major_axis_length * pixel_size_um
            minor_diam_um = prop.minor_axis_length * pixel_size_um

            perim = prop.perimeter
            circularity = (4.0 * np.pi * filled_area) / (perim ** 2 + 1e-6)
            eccentricity = prop.eccentricity

            row = {
                "raw_label": int(prop.label), "cy": round(cy, 1), "cx": round(cx, 1),
                "diam_um": round(diam_um, 2),
                "major_diam_um": round(major_diam_um, 2),
                "minor_diam_um": round(minor_diam_um, 2),
                "circularity": round(circularity, 2),
                "eccentricity": round(eccentricity, 2), "reason": None,
            }

            if (cy < BORDER_MARGIN_PX or cy > h_dim - BORDER_MARGIN_PX or
                    cx < BORDER_MARGIN_PX or cx > w_dim - BORDER_MARGIN_PX):
                row["reason"] = "failed_border"
                debug_list.append(row)
                continue

            if not (MIN_DIAM_UM <= diam_um <= MAX_DIAM_UM):
                row["reason"] = "failed_size"
                debug_list.append(row)
                continue

            if circularity < MIN_CIRCULARITY or eccentricity > MAX_ECCENTRICITY:
                row["reason"] = "failed_shape"
                debug_list.append(row)
                continue

            row["reason"] = "passed"
            debug_list.append(row)

            masks[filled_mask] = idx
            info[idx] = {
                "cy": cy, "cx": cx,
                "diam_um": diam_um,
                "major_diam_um": major_diam_um,
                "minor_diam_um": minor_diam_um,
                "area_um2": filled_area * pixel_area_um2,
                "brightness": float(np.percentile(norm_mip[mask_bool], 90)),
                "circularity": circularity,
                "eccentricity": eccentricity,
                "is_manual": False,
                "raw_label": int(prop.label),
            }
            idx += 1

        return masks, info, idx, debug_list, m

    current_thresh = [default_thresh]
    candidate_masks, candidate_info, next_id, debug_rows, m_label = segment_at_threshold(current_thresh[0])
    active_ids = set(candidate_info.keys())
    next_manual_id = [next_id]

    fig, ax = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
    plt.subplots_adjust(bottom=0.18)
    ax[0].axis("off")
    ax[1].axis("off")

    def get_view():
        display_intensity = norm_mip.copy()
        bg_mask = norm_mip < current_thresh[0]
        display_intensity[bg_mask] *= 0.15

        rgb = np.zeros((h_dim, w_dim, 3), dtype=np.float32)
        rgb[:, :, 0] = display_intensity * base_col[0]
        rgb[:, :, 1] = display_intensity * base_col[1]
        rgb[:, :, 2] = display_intensity * base_col[2]
        overlay = (np.clip(rgb, 0, 1) * 255.0).copy()

        if active_ids:
            valid_mask = np.isin(candidate_masks, np.array(list(active_ids)))
            bounds = find_boundaries(valid_mask, mode="inner")
            bounds = dilation(bounds)
            overlay[bounds] = HIGHLIGHT_COLOR * 255.0

        raw_rgb = np.zeros((h_dim, w_dim, 3), dtype=np.float32)
        raw_rgb[:, :, 0] = norm_mip * base_col[0]
        raw_rgb[:, :, 1] = norm_mip * base_col[1]
        raw_rgb[:, :, 2] = norm_mip * base_col[2]

        return np.clip(overlay, 0, 255).astype(np.uint8), (np.clip(raw_rgb, 0, 1) * 255.0).astype(np.uint8)

    img_raw = ax[0].imshow(np.zeros((h_dim, w_dim, 3), dtype=np.uint8))
    img_eval = ax[1].imshow(np.zeros((h_dim, w_dim, 3), dtype=np.uint8))
    text_objects = []

    def refresh():
        over_img, raw_img = get_view()
        img_raw.set_data(raw_img)
        ax[0].set_title(f"[{file_path.name}] Ch {ch_idx} Raw MIP", fontweight="bold")
        img_eval.set_data(over_img)

        for t in text_objects:
            t.remove()
        text_objects.clear()

        for cid in active_ids:
            info = candidate_info[cid]
            label_text = f"{cid}\n({info['diam_um']:.1f}µm)"
            t = ax[1].text(info["cx"] + 4, info["cy"] - 4, label_text, color="#ffffff", fontsize=7,
                           fontweight="bold",
                           bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.75, edgecolor="#00ffff"))
            text_objects.append(t)

        ax[1].set_title(
            f"Ch {ch_idx} MIP | Thresh: {current_thresh[0]:.2f} | {len(active_ids)} cells\n"
            "[Slider: Adjust | L-Click: Add/Restore | R-Click: Delete | Close: Save & Next]",
            fontweight="bold", fontsize=11)
        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    thresh_slider = Slider(ax_slider, "Threshold", 0.02, 0.80, valinit=default_thresh, valstep=0.01)

    def on_slider(val):
        nonlocal candidate_masks, candidate_info, active_ids, debug_rows, m_label
        current_thresh[0] = float(val)
        candidate_masks, candidate_info, next_manual_id[0], debug_rows, m_label = segment_at_threshold(val)
        active_ids = set(candidate_info.keys())
        refresh()

    thresh_slider.on_changed(on_slider)

    def on_click(event):
        if event.inaxes != ax[1] or event.xdata is None or event.ydata is None:
            return
        cx_full, cy_full = int(round(event.xdata)), int(round(event.ydata))

        hit_id = candidate_masks[cy_full, cx_full]
        if hit_id == 0:
            sub = candidate_masks[max(0, cy_full - 12):min(h_dim, cy_full + 13),
                                  max(0, cx_full - 12):min(w_dim, cx_full + 13)]
            valid = sub[sub > 0]
            if len(valid) > 0:
                counts = np.bincount(valid.flatten())
                counts[0] = 0
                hit_id = counts.argmax()

        if event.button == 1:  # הוספה מקומית חכמה או החזרה
            if hit_id == 0:
                y_min = max(0, cy_full - manual_search_radius_px)
                y_max = min(h_dim, cy_full + manual_search_radius_px + 1)
                x_min = max(0, cx_full - manual_search_radius_px)
                x_max = min(w_dim, cx_full + manual_search_radius_px + 1)

                local_crop = norm_mip[y_min:y_max, x_min:x_max]
                local_thresh = max(0.06, current_thresh[0] * 0.70)
                local_bin = local_crop >= local_thresh
                local_bin = binary_fill_holes(closing(local_bin, disk(2)))
                local_labeled = label(local_bin)

                rel_y, rel_x = cy_full - y_min, cx_full - x_min
                picked_label = (
                    local_labeled[rel_y, rel_x]
                    if (0 <= rel_y < local_bin.shape[0] and 0 <= rel_x < local_bin.shape[1])
                    else 0
                )

                if picked_label > 0:
                    sub_mask = local_labeled == picked_label
                    cell_mask = np.zeros((h_dim, w_dim), dtype=bool)
                    cell_mask[y_min:y_max, x_min:x_max] = sub_mask

                    props = regionprops(sub_mask.astype(int), intensity_image=local_crop)[0]
                    area_px = int(np.sum(sub_mask))
                    diam_um = 2.0 * np.sqrt((area_px * pixel_area_um2) / np.pi)
                    major_diam_um = props.major_axis_length * pixel_size_um
                    minor_diam_um = props.minor_axis_length * pixel_size_um
                    ecc = props.eccentricity
                else:
                    fallback_r = int(round((28.0 / 2.0) / pixel_size_um))  # ברירת מחדל של תא בינוני (~28um)
                    yy, xx = np.ogrid[:h_dim, :w_dim]
                    cell_mask = ((xx - cx_full) ** 2 + (yy - cy_full) ** 2) <= (fallback_r ** 2)
                    area_px = int(np.sum(cell_mask))
                    diam_um = 2.0 * np.sqrt((area_px * pixel_area_um2) / np.pi)
                    major_diam_um = minor_diam_um = diam_um
                    ecc = 0.0

                new_id = next_manual_id[0]
                next_manual_id[0] += 1
                candidate_masks[cell_mask & (candidate_masks == 0)] = new_id

                candidate_info[new_id] = {
                    "cy": cy_full, "cx": cx_full,
                    "diam_um": diam_um,
                    "major_diam_um": major_diam_um,
                    "minor_diam_um": minor_diam_um,
                    "area_um2": area_px * pixel_area_um2,
                    "brightness": float(np.percentile(norm_mip[cell_mask], 90)) if np.any(cell_mask) else 1.0,
                    "circularity": 1.0,
                    "eccentricity": ecc,
                    "is_manual": True,
                    "raw_label": -1,
                }
                active_ids.add(new_id)
                refresh()
            elif hit_id not in active_ids:
                active_ids.add(hit_id)
                refresh()

        elif event.button == 3:  # מחיקה
            if hit_id > 0 and hit_id in active_ids:
                active_ids.discard(hit_id)
                refresh()

    fig.canvas.mpl_connect("button_press_event", on_click)
    refresh()
    plt.show()

    final_valid_ids = set(active_ids)

    if SAVE_DEBUG_OVERLAY and debug_rows:
        _save_debug_overlay(file_path, ch_idx, norm_mip, m_label, debug_rows, h_dim, w_dim)
        pd.DataFrame(debug_rows).to_csv(
            OUTPUT_DIR / f"{prefix}_ch{ch_idx}_all_candidates_debug.csv",
            index=False, encoding="utf-8-sig",
        )

    result_img = np.zeros((h_dim, w_dim, 3), dtype=np.float32)
    result_img[:, :, 0] = norm_mip * base_col[0]
    result_img[:, :, 1] = norm_mip * base_col[1]
    result_img[:, :, 2] = norm_mip * base_col[2]
    result_img = np.clip(result_img * 255.0, 0, 255).astype(np.uint8)
    if final_valid_ids:
        valid_mask = np.isin(candidate_masks, np.array(list(final_valid_ids)))
        bounds = find_boundaries(valid_mask, mode="inner")
        bounds = dilation(bounds)
        result_img[bounds] = HIGHLIGHT_COLOR * 255.0
    cv2.putText(result_img, f"{file_path.stem} Ch{ch_idx}: {len(final_valid_ids)} cells (MIP)",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    result_path = OUTPUT_DIR / f"{prefix}_ch{ch_idx}_MIP_RESULT.png"
    cv2.imwrite(str(result_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print(f"Result image saved to: {result_path} ({len(final_valid_ids)} cells)")

    per_cell_rows = [{
        "parent_folder": file_path.parent.name,
        "file": file_path.name,
        "channel": ch_idx,
        "cell_id": cid,
        "diam_um": round(candidate_info[cid]["diam_um"], 2),
        "major_diam_um": round(candidate_info[cid]["major_diam_um"], 2),
        "minor_diam_um": round(candidate_info[cid]["minor_diam_um"], 2),
        "area_um2": round(candidate_info[cid]["area_um2"], 2),
        "circularity": round(candidate_info[cid]["circularity"], 2),
        "eccentricity": round(candidate_info[cid]["eccentricity"], 2),
        "brightness": round(candidate_info[cid]["brightness"], 4),
        "is_manual": candidate_info[cid]["is_manual"],
        "centroid_y_px": round(candidate_info[cid]["cy"], 1),
        "centroid_x_px": round(candidate_info[cid]["cx"], 1),
    } for cid in final_valid_ids]

    return len(final_valid_ids), per_cell_rows


def _save_debug_overlay(file_path, ch_idx, norm_mip, raw_labels, debug_rows, h_dim, w_dim):
    prefix = f"{file_path.parent.name}_{file_path.stem}"
    base = np.stack([norm_mip, norm_mip, norm_mip], axis=-1)
    overlay = np.clip(base * 255.0, 0, 255).astype(np.uint8).copy()

    for row in debug_rows:
        lbl = row["raw_label"]
        mask_bool = raw_labels == lbl
        if not mask_bool.any():
            continue
        bounds = find_boundaries(mask_bool, mode="inner")
        bounds = dilation(bounds)
        color = DEBUG_COLORS.get(row["reason"], np.array([1, 1, 1]))
        overlay[bounds] = (color * 255).astype(np.uint8)
        cv2.putText(overlay, str(lbl), (int(row["cx"]) + 4, int(row["cy"]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    legend_y = 25
    for reason, color in DEBUG_COLORS.items():
        cv2.putText(overlay, reason, (15, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    tuple(int(c * 255) for c in color[::-1]), 2, cv2.LINE_AA)
        legend_y += 22

    out_path = OUTPUT_DIR / f"{prefix}_ch{ch_idx}_DEBUG_all_candidates.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def process_file(file_path):
    print(f"\n{'#' * 55}")
    print(f"# FILE: {file_path}")
    print(f"{'#' * 55}")

    prefix = f"{file_path.parent.name}_{file_path.stem}"

    with nd2.ND2File(str(file_path)) as f:
        vx = f.voxel_size().x if hasattr(f, "voxel_size") and f.voxel_size().x else None
        vy = f.voxel_size().y if hasattr(f, "voxel_size") and f.voxel_size().y else None
        raw_stack = f.asarray()
        c_axis = list(f.sizes.keys()).index("C") if "C" in f.sizes else None

        channel_colors = {}
        for ch in CHANNELS:
            channel_colors[ch] = get_nd2_channel_color(f, ch)

    if vx is None or vy is None:
        print("  !! WARNING: could not read voxel size from ND2 metadata - falling back to 0.5 um/px.")
        vx = vx or 0.5
        vy = vy or 0.5

    pixel_size_um = float(vx)
    pixel_area_um2 = vx * vy
    print(f"  Using pixel size: {vx:.4f} x {vy:.4f} um/px  (pixel_area_um2 = {pixel_area_um2:.5f})")

    temp_ch = np.take(raw_stack, indices=CHANNELS[0], axis=c_axis) if c_axis is not None else raw_stack
    _, h_dim, w_dim = temp_ch.shape

    all_rows = []
    counts = {}
    for ch in CHANNELS:
        ch_col = channel_colors.get(ch, np.array([1.0, 1.0, 1.0]))
        cnt, rows = process_channel_mip(
            file_path, ch, raw_stack, c_axis,
            pixel_area_um2, pixel_size_um,
            h_dim, w_dim, ch_col
        )
        counts[ch] = cnt
        all_rows.extend(rows)

    print(f"\n{'=' * 55}")
    print(f"SUMMARY - {file_path.name}")
    for ch, cnt in counts.items():
        print(f"  Channel {ch}: {cnt} cells")
    print("=" * 55)

    if all_rows:
        df = pd.DataFrame(all_rows)
        out_csv = OUTPUT_DIR / f"{prefix}_MIP_per_cell.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return all_rows, counts


# ==========================================
# POST-PROCESSING & PLOTTING
# ==========================================
def analyze_and_plot_fiber_types(df, output_dir):
    """
    גרף פרסום עם פרופורציות נעולות – שומר על עמודות צפופות ומאוזנות גם במסך מלא.
    """
    if df.empty or "diam_um" not in df.columns:
        print("No valid cell data available to classify.")
        return df

    # 1. סיווג גודל הסומה ושמירת הנתונים
    df["fiber_type"] = df["diam_um"].apply(classify_soma_size)
    classified_csv = output_dir / "ALL_FILES_classified_cells.csv"
    df.to_csv(classified_csv, index=False, encoding="utf-8-sig")

    # 2. טבלת ספירה
    fiber_order = ["C-fiber", "A-delta", "A-beta/Large"]
    counts_table = df.groupby(["channel", "fiber_type"]).size().unstack(fill_value=0)
    counts_table = counts_table.reindex(columns=fiber_order, fill_value=0)

    channels = list(counts_table.index)
    n_channels = len(channels)

    # 3. הגדרות סגנון
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
    })

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=100)

    colors = {
        "C-fiber": "#2b5c8f",        # Navy
        "A-delta": "#d95f02",        # Terracotta
        "A-beta/Large": "#7570b3"    # Muted Purple
    }

    # קירוב הערוצים זה לזה (0 ו-0.8 במקום 0 ו-1) כדי למנוע חלל ריק באמצע
    cluster_spacing = 0.8
    x_centers = np.arange(n_channels) * cluster_spacing
    bar_width = 0.18
    offsets = [-bar_width * 1.05, 0, bar_width * 1.05]

    for i, fiber in enumerate(fiber_order):
        counts = counts_table[fiber].values
        pos = x_centers + offsets[i]

        bars = ax.bar(
            pos, counts, width=bar_width, label=fiber,
            color=colors[fiber], edgecolor="black", linewidth=0.8, alpha=0.9
        )

        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(
                    f"{int(h)}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color="#111111"
                )

    # נעילת יחס הפרופורציות (Aspect Ratio) של קופסת הגרף
    # מונע מהגרף להימרח לרוחב כאשר מגדילים את החלון
    ax.set_box_aspect(0.85)

    # תחימת ציר ה-X כך שלא יהיו שוליים ריקים ענקיים בצדדים
    left_limit = x_centers[0] - (bar_width * 2.2)
    right_limit = x_centers[-1] + (bar_width * 2.2)
    ax.set_xlim(left_limit, right_limit)

    # ניקוי מסגרות
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="#888888", zorder=0)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([f"Channel {ch}" for ch in channels], fontsize=11, fontweight="semibold")
    ax.set_ylabel("Somata Count", fontsize=11, fontweight="semibold")

    max_val = counts_table.values.max() if len(counts_table.values) > 0 else 5
    ax.set_ylim(0, max_val + max(1.5, max_val * 0.22))

    ax.legend(
        frameon=False,
        loc="upper right",
        fontsize=9,
        borderpad=0.2,
        labelspacing=0.3
    )

    plt.tight_layout()

    out_png = output_dir / "publication_soma_counts.png"
    out_pdf = output_dir / "publication_soma_counts.pdf"
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_pdf, format="pdf")
    plt.show()

    return df
# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\nProcess a single .nd2 file, or a whole folder (recursively)?")
    mode = input("  [f]ile / [d]irectory (default: auto-detect from INPUT_PATH): ").strip().lower()

    if mode in ("f", "file"):
        typed = input(f"  Path to the .nd2 file [{INPUT_PATH}]: ").strip()
        input_path = Path(typed) if typed else Path(INPUT_PATH)
    elif mode in ("d", "dir", "directory"):
        typed = input(f"  Path to the folder [{INPUT_PATH}]: ").strip()
        input_path = Path(typed) if typed else Path(INPUT_PATH)
    else:
        input_path = Path(INPUT_PATH)

    if not input_path.exists():
        print(f"Error: {input_path} not found.")
    else:
        nd2_files = sorted(input_path.rglob("*.nd2")) if input_path.is_dir() else [input_path]

        if not nd2_files:
            print(f"No .nd2 files found under {input_path}.")
        else:
            combined_rows = []
            combined_counts = []

            for fp in nd2_files:
                try:
                    rows, counts = process_file(fp)
                except Exception as e:
                    print(f"  !! Failed to process {fp}: {e}")
                    continue
                combined_rows.extend(rows)
                for ch, cnt in counts.items():
                    combined_counts.append({
                        "parent_folder": fp.parent.name,
                        "file": fp.name,
                        "channel": ch,
                        "cell_count": cnt,
                    })

            print(f"\n{'#' * 55}")
            print("ALL FILES DONE")
            print(f"{'#' * 55}")

            MASTER_SUMMARY_CSV = BASE_OUTPUT_DIR / "ALL_FILES_summary_counts.csv"
            MASTER_PER_CELL_CSV = BASE_OUTPUT_DIR / "ALL_FILES_per_cell.csv"

            if combined_counts:
                summary_df = pd.DataFrame(combined_counts)
                summary_df.to_csv(OUTPUT_DIR / "ALL_FILES_summary_counts.csv", index=False, encoding="utf-8-sig")
                append_to_csv(summary_df, MASTER_SUMMARY_CSV)
                print(f"Appended counts to master: {MASTER_SUMMARY_CSV}")
                print(summary_df.to_string(index=False))

            if combined_rows:
                combined_df = pd.DataFrame(combined_rows)
                combined_df.to_csv(OUTPUT_DIR / "ALL_FILES_per_cell.csv", index=False, encoding="utf-8-sig")
                append_to_csv(combined_df, MASTER_PER_CELL_CSV)
                print(f"Appended cell data to master: {MASTER_PER_CELL_CSV}")

                # סיווג סיבים והפקת הגרפים
                analyze_and_plot_fiber_types(combined_df, OUTPUT_DIR)