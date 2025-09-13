# App.py — v8 (doc-aligned CLI)

from pathlib import Path
import argparse
from adaptive_paddleocr import adaptive_ocr, simple_ocr, build_ocr

def main():
    p = argparse.ArgumentParser(description="Adaptive / Simple PaddleOCR app (v8)")
    p.add_argument("image", nargs="?", type=Path, default=Path("napoleaomendes.png"),
                   help="Image file (default: napoleaomendes.png)")

    # flags corretos
    p.add_argument("--simple", "-S", action="store_true", help="Run simple mode")
    p.add_argument("--save", "-s", action="store_true", help="Save PNG/JSON outputs")

    # modelos/idiomas
    p.add_argument("--lang", default="en", help="OCR language (PP-OCRv5 supports la, pt, etc.)")
    p.add_argument("--rec_model", default=None, help="Recognition model name (if set, lang is ignored)")
    p.add_argument("--det_model", default=None, help="Detection model name (if set, lang is ignored)")

    # parâmetros de predict() (doc v3.x)
    p.add_argument("--text_rec_score_thresh", type=float, default=0.0,
                   help="Keep texts with score >= this (default 0.0 = no filter)")
    p.add_argument("--use_doc_unwarping", type=lambda x: x.lower() == "true",
                   default=False, help="Enable document unwarping")
    p.add_argument("--use_textline_orientation", type=lambda x: x.lower() == "true",
                   default=False, help="Enable textline orientation module")
    p.add_argument("--return_word_box", type=lambda x: x.lower()=="true", default=False,
               help="Se True, retorna caixas em nível de palavra do recognizer.")


    args = p.parse_args()
    ocr = build_ocr(args)

    if args.simple:
        print("Running SIMPLE mode…")
        txt = simple_ocr(ocr, args.image, args.save, args=args)
        print("\n--- TEXT ---\n", txt)
    else:
        print("Running ADAPTIVE mode…")
        txt, step = adaptive_ocr(ocr, args.image, args.save, args=args)
        print(f"\n✅ Best step: {step}\n{txt}")

if __name__ == "__main__":
    main()
