"""
Convert all PDFs in a directory to JPG images (one image per page).

Usage:
  python src/convert_pdfs_to_images.py --input <pdf_dir> --output <output_dir> [--dpi 300]
"""

import argparse
from pathlib import Path
from pdf2image import convert_from_path


def convert_pdfs(input_dir: Path, output_dir: Path, dpi: int = 300):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDFs, converting at {dpi} DPI...\n")

    total = 0
    for pdf_path in pdf_files:
        pages = convert_from_path(str(pdf_path), dpi=dpi)
        stem = pdf_path.stem

        for i, page in enumerate(pages):
            if len(pages) == 1:
                filename = f"{stem}.jpg"
            else:
                filename = f"{stem}_page-{i+1}.jpg"

            out_path = output_dir / filename
            page.save(str(out_path), "JPEG", quality=95)
            total += 1

        print(f"  {pdf_path.name} → {len(pages)} page(s)")

    print(f"\nDone! {total} images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDFs to JPG images")
    parser.add_argument("--input", type=str, required=True, help="Directory with PDF files")
    parser.add_argument("--output", type=str, required=True, help="Output directory for JPG images")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution (default: 300)")
    args = parser.parse_args()

    convert_pdfs(Path(args.input), Path(args.output), args.dpi)
