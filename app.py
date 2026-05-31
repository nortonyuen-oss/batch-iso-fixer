from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st
from PIL import Image

from image_processor import (
    ProcessingSettings,
    list_images,
    list_images_recursive,
    load_image,
    process_image,
    process_batch,
    process_recursive_batch,
    make_before_after,
    remove_background,
    crop_transparent,
    scale_image,
    detect_base_angle,
)

st.set_page_config(page_title="Batch Isometric Asset Fixer", layout="wide")

st.title("Batch Isometric Asset Fixer")
st.caption("批量去背、裁切、X/Y比例修正、透明 PNG 輸出。適合 isometric building / tile assets。")


def composite_on_color(image: Image.Image, color: tuple[int, int, int]) -> Image.Image:
    img = image.convert("RGBA")
    bg = Image.new("RGBA", img.size, (*color, 255))
    bg.alpha_composite(img, (0, 0))
    return bg.convert("RGB")


if "settings" not in st.session_state:
    st.session_state.settings = ProcessingSettings()

settings: ProcessingSettings = st.session_state.settings

with st.sidebar:
    st.header("Folders")
    input_folder = st.text_input("Input folder", value="input")
    output_folder = st.text_input("Output folder", value="output/fixed")
    raw_input_folder = st.text_input("Raw tree input", value="input/raw")
    modified_output_folder = st.text_input("Modified tree output", value="output/modified")

    st.divider()
    st.header("Background")
    settings.skip_background_removal = st.checkbox(
        "Skip background removal",
        value=settings.skip_background_removal,
        help="圖片已完成去背（帶透明通道），直接跳過去背步驟，只做比例修正。",
    )
    if not settings.skip_background_removal:
        settings.background_mode = st.selectbox(
            "Background mode",
            options=["auto_corner", "white", "custom"],
            index=["auto_corner", "white", "custom"].index(settings.background_mode),
            help="auto_corner 會用圖片四角估算背景色。白底圖通常用 auto_corner 或 white 都可以。",
        )
        settings.tolerance = st.slider("Background tolerance", 0, 100, settings.tolerance)
        if settings.background_mode == "custom":
            settings.custom_bg_hex = st.color_picker("Custom background color", value=settings.custom_bg_hex)
        settings.feather_edges = st.checkbox("Feather alpha edges", value=settings.feather_edges)
        if settings.feather_edges:
            settings.feather_radius = st.slider("Feather radius", 0.0, 3.0, float(settings.feather_radius), 0.1)

    st.divider()
    st.header("Anti-halo")
    settings.defringe_edges = st.checkbox(
        "Remove white matte / defringe",
        value=settings.defringe_edges,
        help="修正半透明邊緣入面殘留嘅白色 RGB，減少遊戲內白邊。",
    )
    settings.edge_bleed_pixels = st.slider(
        "RGB bleed pixels",
        0,
        8,
        int(settings.edge_bleed_pixels),
        help="將可見像素顏色向透明區擴展，避免 GPU/縮放 sample 到白色透明像素。",
    )

    st.divider()
    st.header("Crop / Scale")
    settings.crop_transparent = st.checkbox("Crop transparent border", value=settings.crop_transparent)
    settings.crop_padding = st.number_input("Crop padding", min_value=0, max_value=300, value=settings.crop_padding, step=1)
    settings.scale_x = st.number_input("Scale X", min_value=0.1, max_value=3.0, value=float(settings.scale_x), step=0.01, format="%.2f")
    settings.scale_y = st.number_input("Scale Y", min_value=0.1, max_value=3.0, value=float(settings.scale_y), step=0.01, format="%.2f")
    settings.align_to_isometric = st.checkbox("Align figure to 1:2 isometric scale", value=settings.align_to_isometric)
    if settings.align_to_isometric:
        settings.iso_target_ratio = st.number_input(
            "Target isometric Y/X",
            min_value=0.1,
            max_value=2.0,
            value=float(settings.iso_target_ratio),
            step=0.01,
            format="%.2f",
        )
        settings.iso_max_side_diff = st.number_input(
            "Max left/right ratio difference",
            min_value=0.0,
            max_value=2.0,
            value=float(settings.iso_max_side_diff),
            step=0.01,
            format="%.2f",
        )
        settings.iso_use_virtual_bottom_corner = st.checkbox(
            "Use virtual bottom corner for chopped bases",
            value=settings.iso_use_virtual_bottom_corner,
        )

    st.divider()
    st.header("Canvas")
    settings.canvas_width = st.number_input("Canvas width", min_value=64, max_value=4096, value=settings.canvas_width, step=64)
    settings.canvas_height = st.number_input("Canvas height", min_value=64, max_value=4096, value=settings.canvas_height, step=64)
    settings.alignment = st.selectbox(
        "Alignment",
        options=["bottom-center", "center"],
        index=["bottom-center", "center"].index(settings.alignment),
    )
    settings.bottom_margin = st.number_input("Bottom margin", min_value=-1000, max_value=1000, value=settings.bottom_margin, step=1)
    settings.preserve_if_too_large = st.checkbox("Do not auto-fit if image larger than canvas", value=settings.preserve_if_too_large)

    st.divider()
    st.header("Settings JSON")
    st.download_button("Download settings.json", data=settings.to_json(), file_name="settings.json", mime="application/json")
    uploaded_settings = st.file_uploader("Load settings.json", type=["json"])
    if uploaded_settings is not None:
        try:
            st.session_state.settings = ProcessingSettings.from_json(uploaded_settings.read().decode("utf-8"))
            st.success("Settings loaded. Please rerun / interact once to refresh UI.")
        except Exception as exc:
            st.error(f"Cannot load settings: {exc}")

st.session_state.settings = settings

input_path = Path(input_folder).expanduser()
output_path = Path(output_folder).expanduser()
images = list_images(input_path)
raw_input_path = Path(raw_input_folder).expanduser()
modified_output_path = Path(modified_output_folder).expanduser()
raw_images = list_images_recursive(raw_input_path)

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Images found", len(images))
col_b.metric("Scale X", f"{settings.scale_x:.2f}")
col_c.metric("Scale Y", f"{settings.scale_y:.2f}")
col_d.metric("Canvas", f"{settings.canvas_width}×{settings.canvas_height}")

if not input_path.exists():
    st.warning(f"Input folder does not exist yet: {input_path.resolve()}")
    st.info("你可以先建立 input/，放入 PNG/JPG/WEBP，再 refresh。")
elif not images:
    st.warning("No images found in input folder.")
else:
    st.subheader("Image list")
    with st.expander("Show files", expanded=False):
        st.write([p.name for p in images])

    st.subheader("Preview")
    preview_count = st.slider("Preview count", 1, min(10, len(images)), min(3, len(images)))
    analyze_angle = st.checkbox("Try OpenCV base angle analysis", value=False, help="實驗功能：嘗試估底部斜線角度，AI 圖陰影太多可能不準。")

    for path in images[:preview_count]:
        st.markdown(f"### {path.name}")
        try:
            before = load_image(path)
            after = process_image(before, settings)
            c1, c2, c3 = st.columns([1, 1, 1])
            c1.image(before, caption="Before", use_container_width=True)
            c2.image(after, caption="After", use_container_width=True)
            combined = make_before_after(before, after)
            c3.image(combined, caption="Before / After", use_container_width=True)

            with st.expander("Halo debug backgrounds", expanded=False):
                bg1, bg2, bg3 = st.columns(3)
                bg1.image(composite_on_color(after, (20, 20, 20)), caption="Dark", use_container_width=True)
                bg2.image(composite_on_color(after, (70, 170, 65)), caption="Game green", use_container_width=True)
                bg3.image(composite_on_color(after, (120, 120, 120)), caption="Road gray", use_container_width=True)

            if analyze_angle:
                if settings.skip_background_removal:
                    temp = before.convert("RGBA")
                else:
                    temp = remove_background(before, settings.background_mode, settings.tolerance, settings.custom_bg_hex, settings.feather_edges, settings.feather_radius)
                if settings.crop_transparent:
                    temp = crop_transparent(temp, settings.crop_padding)
                temp = scale_image(temp, settings.scale_x, settings.scale_y)
                angle, suggested = detect_base_angle(temp)
                if angle is None:
                    st.caption("Angle: not detected")
                else:
                    st.caption(f"Detected base angle: {angle}° | suggested scale_y: {suggested}")
        except Exception as exc:
            st.error(f"Preview failed for {path.name}: {exc}")

    st.divider()
    st.subheader("Batch Export")
    st.write(f"Output folder: `{output_path.resolve()}`")

    if st.button("Process all images", type="primary"):
        with st.spinner("Processing images..."):
            results = process_batch(input_path, output_path, settings, analyze_angle=analyze_angle)
        df = pd.DataFrame([r.__dict__ for r in results])
        success = int((df["status"] == "success").sum()) if not df.empty else 0
        failed = len(df) - success
        st.success(f"Done. Success: {success}, Failed: {failed}")
        st.dataframe(df, use_container_width=True)

        log_path = output_path / "process_log.csv"
        settings_path = output_path / "settings.json"
        if log_path.exists():
            st.download_button("Download process_log.csv", data=log_path.read_bytes(), file_name="process_log.csv", mime="text/csv")
        if settings_path.exists():
            st.download_button("Download exported settings.json", data=settings_path.read_bytes(), file_name="settings.json", mime="application/json")

st.divider()
st.subheader("Recursive Raw Folder Export")
st.caption("一按處理 `input/raw/` 內所有子資料夾圖片，並喺 `output/modified/` 重現相同資料夾結構。")

rc1, rc2, rc3 = st.columns(3)
rc1.metric("Raw tree images", len(raw_images))
rc2.write(f"Input: `{raw_input_path.resolve()}`")
rc3.write(f"Output: `{modified_output_path.resolve()}`")

if not raw_input_path.exists():
    st.warning(f"Raw input folder does not exist yet: {raw_input_path.resolve()}")
elif not raw_images:
    st.warning("No recursive images found in raw input folder.")
else:
    with st.expander("Show raw tree files", expanded=False):
        st.write([p.relative_to(raw_input_path).as_posix() for p in raw_images])

    if st.button("Process raw folder tree", type="primary"):
        with st.spinner("Processing full raw folder tree..."):
            results = process_recursive_batch(raw_input_path, modified_output_path, settings, analyze_angle=False)
        df = pd.DataFrame([r.__dict__ for r in results])
        success = int((df["status"] == "success").sum()) if not df.empty else 0
        failed = len(df) - success
        st.success(f"Done. Success: {success}, Failed: {failed}")
        st.dataframe(df, use_container_width=True)

        log_path = modified_output_path / "process_log.csv"
        settings_path = modified_output_path / "settings.json"
        if log_path.exists():
            st.download_button("Download recursive process_log.csv", data=log_path.read_bytes(), file_name="process_log.csv", mime="text/csv")
        if settings_path.exists():
            st.download_button("Download recursive settings.json", data=settings_path.read_bytes(), file_name="settings.json", mime="application/json")

st.divider()
st.subheader("One-off Upload Test")
st.caption("未想整理 folder 時，可以先拖一張圖入嚟試效果。")
uploaded = st.file_uploader("Upload single image", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGBA")
    after = process_image(img, settings)
    cc1, cc2 = st.columns(2)
    cc1.image(img, caption="Uploaded", use_container_width=True)
    cc2.image(after, caption="Processed", use_container_width=True)

    with st.expander("Halo debug backgrounds", expanded=True):
        bg1, bg2, bg3 = st.columns(3)
        bg1.image(composite_on_color(after, (20, 20, 20)), caption="Dark", use_container_width=True)
        bg2.image(composite_on_color(after, (70, 170, 65)), caption="Game green", use_container_width=True)
        bg3.image(composite_on_color(after, (120, 120, 120)), caption="Road gray", use_container_width=True)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        after.save(tmp.name)
        data = Path(tmp.name).read_bytes()
    st.download_button("Download processed PNG", data=data, file_name=f"{Path(uploaded.name).stem}_fixed.png", mime="image/png")
