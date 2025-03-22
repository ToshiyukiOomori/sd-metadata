import io
import re
import pandas as pd
import streamlit as st
from PIL import Image, PngImagePlugin, WebPImagePlugin
import piexif
import piexif.helper

LABELS = [
    "Prompt:",
    "Negative prompt:",
    "Steps:",
    "Sampler:",
    "CFG scale:",
    "Seed:",
    "Size:",
    "Model hash:",
    "Model:",
    "Version:"
]

RGX_LABEL = re.compile(
    r"(?P<label>(?:Prompt|Negative prompt|Steps|Sampler|CFG scale|Seed|Size|Model hash|Model|Version)):\s*"
    r"(?P<value>.*?)(?=\s*(?:Prompt:|Negative prompt:|Steps:|Sampler:|CFG scale:|Seed:|Size:|Model hash:|Model|Version|$))",
    flags=re.IGNORECASE | re.DOTALL
)

def parse_labeled(text: str) -> dict:
    results = {}
    for m in RGX_LABEL.finditer(text):
        label_raw = m.group("label").strip()
        value_raw = m.group("value").strip()
        key_lower = label_raw.lower()
        if key_lower == "cfg scale":
            results["CFG Scale"] = value_raw
        elif key_lower == "negative prompt":
            results["Negative Prompt"] = value_raw
        elif key_lower == "model hash":
            results["Model hash"] = value_raw
        else:
            # 例: Prompt -> Prompt, Steps -> Steps, etc
            results[label_raw.title()] = value_raw
    return results

def parse_no_prompt_label(raw_text: str) -> dict:
    """
    テキストに「Prompt:」ラベルが無い場合、冒頭から次のラベル出現までを Prompt とみなす
    """
    pattern_next_label = r"(?=(" + "|".join(map(re.escape, LABELS[1:])) + r") )"  # Negative prompt:等
    match = re.search(pattern_next_label, raw_text, flags=re.IGNORECASE)
    if not match:
        return {"Prompt": raw_text.strip()}
    start_index = 0
    end_index = match.start()
    prompt_text = raw_text[start_index:end_index].strip()
    rest_text = raw_text[end_index:].strip()
    d = {}
    if prompt_text:
        d["Prompt"] = prompt_text
    labeled = parse_labeled(rest_text)
    d.update(labeled)
    return d

def parse_sd_text(raw_text: str) -> dict:
    data = parse_labeled(raw_text)
    if "Prompt" not in [k.lower() for k in data.keys()]:
        data = parse_no_prompt_label(raw_text)
    data["RawSnippet"] = raw_text
    return data

def fallback_search(bdata: bytes) -> str:
    return bdata.decode("utf-8", errors="ignore")

def extract_exif_usercomment(img):
    if "exif" in img.info:
        try:
            exif_dict = piexif.load(img.info["exif"])
            uc = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, None)
            if uc:
                try:
                    return piexif.helper.UserComment.load(uc)
                except:
                    return uc.decode("utf-8", errors="ignore")
        except:
            pass
    return ""

def extract_sd_meta(bdata: bytes) -> dict:
    meta_info = {}
    try:
        with Image.open(io.BytesIO(bdata)) as img:
            meta_info["format"] = img.format
            details = {}
            if img.format in ("PNG","WEBP"):
                details.update(img.info)
            if isinstance(img, PngImagePlugin.PngImageFile):
                details.update(img.text)
            if isinstance(img, WebPImagePlugin.WebPImageFile) and hasattr(img, "text"):
                details.update(img.text)

            pillow_str = details.get("parameters", "")
            exif_str = ""
            if img.format == "WEBP":
                exif_str = extract_exif_usercomment(img)

            if pillow_str:
                meta_info["foundIn"] = "Pillow info/text"
                parsed = parse_sd_text(pillow_str)
                meta_info.update(parsed)
            elif exif_str:
                meta_info["foundIn"] = "Exif UserComment"
                parsed = parse_sd_text(exif_str)
                meta_info.update(parsed)
            else:
                meta_info["foundIn"] = "Fallback"
                fb = fallback_search(bdata)
                parsed = parse_sd_text(fb)
                meta_info.update(parsed)
    except Exception as e:
        meta_info["error"] = str(e)
    return meta_info

def main():
    st.set_page_config(page_title="SD Metadata Viewer", layout="wide")
    st.title("Stable Diffusion Metadata Viewer (PNG/WEBP)")

    # 複数ファイルをドラッグ＆ドロップ
    uploaded_files = st.file_uploader(
        "Drag & drop or click to upload multiple files",
        type=["png","webp"],
        accept_multiple_files=True,
        label_visibility="visible"
    )

    if uploaded_files:
        # ファイル名リスト
        file_names = [f.name for f in uploaded_files]

        # ラジオボタンで選択
        selected_file_name = st.radio(
            "Select an image to view parameters",
            file_names,
            label_visibility="visible"
        )

        # 選択されたファイルを探す
        chosen = next((x for x in uploaded_files if x.name == selected_file_name), None)
        if chosen is not None:
            bdata = chosen.read()
            try:
                img = Image.open(io.BytesIO(bdata))
                col1, col2 = st.columns([1,1], gap="medium")
                with col1:
                    st.image(img, caption=f"{selected_file_name}", use_container_width=True)
                with col2:
                    meta = extract_sd_meta(bdata)
                    if "error" in meta:
                        st.error(meta["error"])
                    else:
                        if meta.get("foundIn") == "Fallback":
                            st.warning("Metadata from fallback scan.")
                        elif meta.get("foundIn") is None:
                            st.warning("No SD metadata found.")
                        else:
                            st.markdown(f"**Data Found In:** {meta.get('foundIn','Unknown')}")

                        raw_txt = meta.pop("RawSnippet", None)
                        meta.pop("foundIn", None)
                        meta.pop("format", None)

                        if len(meta) > 0:
                            order = ["Prompt","Negative Prompt","Steps","Sampler",
                                     "CFG Scale","Seed","Size","Model hash","Model","Version"]
                            display_dict = {}
                            for k in order:
                                for mk in list(meta.keys()):
                                    if mk.lower() == k.lower():
                                        display_dict[k] = meta.pop(mk)
                                        break
                            display_dict.update(meta)

                            df = pd.DataFrame(list(display_dict.items()), columns=["Key","Value"])
                            st.table(df)
                        else:
                            st.info("No recognized parameters.")

                        # ラベルは与えつつ、見た目は折りたたむ
                        if raw_txt:
                            with st.expander("Raw SD Text"):
                                st.text_area(
                                    "Raw Data",
                                    raw_txt,
                                    height=200,
                                    label_visibility="collapsed"
                                )
            except Exception as e:
                st.error(f"Cannot open image: {e}")

if __name__ == "__main__":
    main()
