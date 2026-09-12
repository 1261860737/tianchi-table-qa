"""PDF/图片统一转为适合视觉模型输入的 JPEG 页面。"""

from __future__ import annotations

import base64
import hashlib
import threading
from pathlib import Path

import pymupdf
from PIL import Image, ImageOps

from table_qa_agent.config import DocumentConfig
from table_qa_agent.schemas import RegionRef

SUPPORTED_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class DocumentProcessingError(RuntimeError):
    """文档无法转成模型输入。"""


class DocumentProcessor:
    """带内容指纹缓存的文档预处理器。"""

    def __init__(self, config: DocumentConfig) -> None:
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self._prepare_lock = threading.Lock()

    def prepare(self, source: Path | str) -> list[Path]:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"文档不存在: {path}")

        cache_dir = self.config.cache_dir / self._cache_key(path)
        marker = cache_dir / ".complete"
        cached_pages = sorted(cache_dir.glob("page-*.jpg"))
        if marker.exists() and cached_pages:
            return cached_pages

        # 多道题可能同时引用同一文档，避免首次缓存时并发写同一个文件。
        with self._prepare_lock:
            cached_pages = sorted(cache_dir.glob("page-*.jpg"))
            if marker.exists() and cached_pages:
                return cached_pages

            cache_dir.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".pdf":
                pages = self._render_pdf(path, cache_dir)
            elif path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                output = cache_dir / "page-001.jpg"
                self._save_normalized_image(path, output)
                pages = [output]
            else:
                raise DocumentProcessingError(f"不支持的文档格式: {path.suffix}")

            marker.touch()
            return pages

    def _cache_key(self, path: Path) -> str:
        stat = path.stat()
        payload = (
            f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|"
            f"{self.config.pdf_dpi}|{self.config.max_long_edge}|"
            f"{self.config.jpeg_quality}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def _render_pdf(self, source: Path, cache_dir: Path) -> list[Path]:
        try:
            document = pymupdf.open(source)
        except Exception as exc:
            raise DocumentProcessingError(f"PDF 打开失败: {source}") from exc

        with document:
            if document.page_count == 0:
                raise DocumentProcessingError(f"PDF 没有页面: {source}")
            if document.page_count > self.config.max_pages:
                raise DocumentProcessingError(
                    f"PDF 共 {document.page_count} 页，超过 max_pages={self.config.max_pages}"
                )

            zoom = self.config.pdf_dpi / 72.0
            matrix = pymupdf.Matrix(zoom, zoom)
            outputs: list[Path] = []
            for index, page in enumerate(document):
                pixmap = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)
                image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                output = cache_dir / f"page-{index + 1:03d}.jpg"
                self._save_pillow_image(image, output)
                outputs.append(output)
            return outputs

    def _save_normalized_image(self, source: Path, output: Path) -> None:
        try:
            with Image.open(source) as image:
                normalized = ImageOps.exif_transpose(image).convert("RGB")
                self._save_pillow_image(normalized, output)
        except Exception as exc:
            raise DocumentProcessingError(f"图片读取失败: {source}") from exc

    def _save_pillow_image(self, image: Image.Image, output: Path) -> None:
        width, height = image.size
        long_edge = max(width, height)
        if long_edge > self.config.max_long_edge:
            scale = self.config.max_long_edge / long_edge
            image = image.resize(
                (max(1, round(width * scale)), max(1, round(height * scale))),
                Image.Resampling.LANCZOS,
            )
        image.save(
            output,
            format="JPEG",
            quality=self.config.jpeg_quality,
            optimize=True,
            progressive=True,
        )

    def to_data_url(self, page: Path) -> str:
        raw = page.read_bytes()
        encoded = base64.b64encode(raw)
        if len(encoded) > self.config.max_encoded_bytes:
            raise DocumentProcessingError(
                f"页面 Base64 大小 {len(encoded)} 字节超过限制 "
                f"{self.config.max_encoded_bytes}: {page}"
            )
        return f"data:image/jpeg;base64,{encoded.decode('ascii')}"

    def as_openai_content(self, pages: list[Path]) -> list[dict[str, object]]:
        """按页码构造 OpenAI Chat Completions 多模态 content。"""

        content: list[dict[str, object]] = []
        for page_number, page in enumerate(pages, start=1):
            content.append({"type": "text", "text": f"文档第 {page_number} 页："})
            image_url: dict[str, object] = {"url": self.to_data_url(page)}
            content.append({"type": "image_url", "image_url": image_url})
        return content

    def as_region_content(
        self,
        regions: list[Path],
        references: list[RegionRef],
    ) -> list[dict[str, object]]:
        if len(regions) != len(references):
            raise ValueError("候选区域文件与 RegionRef 数量不一致")
        content: list[dict[str, object]] = []
        for index, (region, reference) in enumerate(zip(regions, references, strict=True), start=1):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"候选区域 {index}，来自原文档第 {reference.page} 页，"
                        f"已顺时针旋转 {reference.rotation_degrees} 度以正向阅读；"
                        "逻辑行列按正向表格解释，不把区域图当作额外表格："
                    ),
                }
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self.to_data_url(region)},
                }
            )
        return content

    def prepare_regions(
        self,
        source: Path | str,
        regions: list[RegionRef],
        *,
        roi_dpi: int = 360,
        padding_ratio: float = 0.03,
    ) -> list[Path]:
        """从原始文件生成高清候选区域，避免对低清缓存图片直接放大。"""

        path = Path(source)
        outputs: list[Path] = []
        for index, region in enumerate(regions, start=1):
            padded = self._padded_bbox(region.bbox, padding_ratio)
            token = hashlib.sha256(
                (
                    f"{path.resolve()}|{path.stat().st_mtime_ns}|{region.page}|{padded}|"
                    f"{roi_dpi}|rotation={region.rotation_degrees}"
                ).encode()
            ).hexdigest()[:20]
            output_dir = self.config.cache_dir / "regions" / token
            output = output_dir / f"region-{index:03d}.jpg"
            if output.is_file():
                outputs.append(output)
                continue
            with self._prepare_lock:
                if output.is_file():
                    outputs.append(output)
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                if path.suffix.lower() == ".pdf":
                    self._render_pdf_region(path, region.page, padded, roi_dpi, output)
                elif path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                    self._render_image_region(path, padded, output)
                else:
                    raise DocumentProcessingError(f"不支持的文档格式: {path.suffix}")
                if region.rotation_degrees:
                    with Image.open(output) as rendered:
                        rotated = rendered.rotate(-region.rotation_degrees, expand=True)
                    self._save_pillow_image(rotated, output)
            outputs.append(output)
        return outputs

    @staticmethod
    def _padded_bbox(
        bbox: tuple[float, float, float, float], padding_ratio: float
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        return (
            max(0.0, x1 - width * padding_ratio),
            max(0.0, y1 - height * padding_ratio),
            min(1.0, x2 + width * padding_ratio),
            min(1.0, y2 + height * padding_ratio),
        )

    def _render_pdf_region(
        self,
        source: Path,
        page_number: int,
        bbox: tuple[float, float, float, float],
        dpi: int,
        output: Path,
    ) -> None:
        with pymupdf.open(source) as document:
            if not 1 <= page_number <= document.page_count:
                raise DocumentProcessingError(f"PDF 页码越界: {page_number}")
            page = document[page_number - 1]
            x1, y1, x2, y2 = bbox
            page_rect = page.rect
            clip = pymupdf.Rect(
                page_rect.x0 + page_rect.width * x1,
                page_rect.y0 + page_rect.height * y1,
                page_rect.x0 + page_rect.width * x2,
                page_rect.y0 + page_rect.height * y2,
            )
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(dpi / 72.0, dpi / 72.0),
                clip=clip,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            self._save_pillow_image(image, output)

    def _render_image_region(
        self,
        source: Path,
        bbox: tuple[float, float, float, float],
        output: Path,
    ) -> None:
        with Image.open(source) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            width, height = normalized.size
            x1, y1, x2, y2 = bbox
            crop_box = (
                round(width * x1),
                round(height * y1),
                max(round(width * x1) + 1, round(width * x2)),
                max(round(height * y1) + 1, round(height * y2)),
            )
            self._save_pillow_image(normalized.crop(crop_box), output)
