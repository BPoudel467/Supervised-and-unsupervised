from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from src.utils.common import FIGURES_DIR, MODELS_DIR, TABLES_DIR, ensure_directories


@dataclass(frozen=True)
class FileItem:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def size_bytes(self) -> int:
        try:
            return self.path.stat().st_size
        except FileNotFoundError:
            return 0

    @property
    def mtime_ns(self) -> int:
        try:
            return self.path.stat().st_mtime_ns
        except FileNotFoundError:
            return 0


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _list_files(directory: Path, exts: set[str] | None = None) -> list[FileItem]:
    if not directory.exists():
        return []
    paths: Iterable[Path] = directory.iterdir()
    items: list[FileItem] = []
    for p in paths:
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        items.append(FileItem(p))
    return items


def _human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n = n / 1024
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"


def _download_button(item: FileItem, key: str) -> None:
    try:
        data = item.path.read_bytes()
    except FileNotFoundError:
        st.warning(f"File missing: {item.name}")
        return
    st.download_button(
        label=f"Download {item.name}",
        data=data,
        file_name=item.name,
        mime="application/octet-stream",
        key=key,
        use_container_width=True,
    )


def _render_figures() -> None:
    st.subheader("Figures")
    figures = _list_files(FIGURES_DIR, IMAGE_EXTS)

    if not figures:
        st.info(f"No figure images found in `{FIGURES_DIR}` yet.")
        st.caption("Run the pipeline (`python main.py`) to generate outputs.")
        return

    sort_mode = st.selectbox("Sort by", ["Newest first", "Oldest first", "Name (A→Z)", "Name (Z→A)"])
    if sort_mode == "Newest first":
        figures.sort(key=lambda x: x.mtime_ns, reverse=True)
    elif sort_mode == "Oldest first":
        figures.sort(key=lambda x: x.mtime_ns)
    elif sort_mode == "Name (A→Z)":
        figures.sort(key=lambda x: x.name.lower())
    else:
        figures.sort(key=lambda x: x.name.lower(), reverse=True)

    cols = st.slider("Columns", min_value=1, max_value=4, value=2)
    grid = st.columns(cols)

    for i, item in enumerate(figures):
        with grid[i % cols]:
            st.image(str(item.path), caption=item.name, use_container_width=True)
            st.caption(f"{_human_bytes(item.size_bytes)}")
            _download_button(item, key=f"dl_fig_{item.name}_{i}")


def _render_tables() -> None:
    st.subheader("Tables")
    tables = _list_files(TABLES_DIR, {".csv"})

    if not tables:
        st.info(f"No CSV tables found in `{TABLES_DIR}` yet.")
        st.caption("Run the pipeline (`python main.py`) to generate outputs.")
        return

    tables.sort(key=lambda x: x.name.lower())
    selected = st.selectbox("Select a table", [t.name for t in tables])
    item = next(t for t in tables if t.name == selected)

    try:
        df = pd.read_csv(item.path)
    except Exception as e:
        st.error(f"Failed to read `{item.name}`: {e}")
        _download_button(item, key=f"dl_tbl_raw_{item.name}")
        return

    st.caption(f"{item.name} • {df.shape[0]} rows × {df.shape[1]} cols • {_human_bytes(item.size_bytes)}")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download {item.name}",
        data=csv_bytes,
        file_name=item.name,
        mime="text/csv",
        use_container_width=True,
    )


def _render_models() -> None:
    st.subheader("Models")
    models = _list_files(MODELS_DIR)

    if not models:
        st.info(f"No model artifacts found in `{MODELS_DIR}` yet.")
        st.caption("Run the pipeline (`python main.py`) to generate outputs.")
        return

    models.sort(key=lambda x: x.name.lower())
    for i, item in enumerate(models):
        with st.expander(item.name, expanded=False):
            st.write(f"Size: **{_human_bytes(item.size_bytes)}**")
            _download_button(item, key=f"dl_model_{item.name}_{i}")


def main() -> None:
    ensure_directories()

    st.set_page_config(page_title="IDS Outputs Dashboard", layout="wide")
    st.title("IDS Outputs Dashboard")
    st.caption("Auto-discovers outputs in `outputs/figures`, `outputs/tables`, and `outputs/models`.")

    tab_fig, tab_tbl, tab_models = st.tabs(["Figures", "Tables", "Models"])
    with tab_fig:
        _render_figures()
    with tab_tbl:
        _render_tables()
    with tab_models:
        _render_models()


if __name__ == "__main__":
    main()

