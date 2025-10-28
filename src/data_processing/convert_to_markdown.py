from pathlib import Path
import argparse

import docling
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions


def clean_filename(name):
    """Sanitize filename by replacing invalid characters"""
    safe = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in name)
    return "_".join(filter(None, safe.split("_")))


def convert_pdfs(input_dir, output_dir, use_cuda=False, num_threads=8, max_pages=10):
    """Convert PDF files to Markdown format"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    docling.utils.model_downloader.download_models()
    
    pipeline_options = PdfPipelineOptions()
    device = AcceleratorDevice.CUDA if use_cuda else AcceleratorDevice.CPU
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=num_threads, device=device)
    pipeline_options.do_code_enrichment = True
    pipeline_options.do_formula_enrichment = True
    
    format_options = {InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    converter = DocumentConverter(format_options=format_options)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdfs = sorted(input_dir.rglob("*.pdf"))
    
    for i, pdf_path in enumerate(pdfs, 1):
        doc_id = clean_filename(f"{pdf_path.stem}")
        
        try:
            result = converter.convert(
                str(pdf_path), 
                page_range=(1, max_pages), 
                max_file_size=10485760, 
                raises_on_error=False
            )
            
            if result.status != ConversionStatus.SUCCESS:
                print(f"[{i}/{len(pdfs)}] FAILED {pdf_path.name}: {result.status}")
                continue
                
            doc = result.document
            markdown_content = doc.export_to_markdown()
            (output_dir / f"{doc_id}.md").write_text(markdown_content, encoding="utf-8")
            
            print(f"[{i}/{len(pdfs)}] OK  {pdf_path.name} -> {doc_id}.md")
            
        except Exception as e:
            print(f"[{i}/{len(pdfs)}] ERROR {pdf_path.name}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="Convert PDF files to Markdown")
    parser.add_argument("--input-dir", default='data/pdf", help="Input directory with PDFs")
    parser.add_argument("--output-dir", default='data/converted_md", help="Output directory")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA acceleration")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--max-pages", type=int, default=10, help="Max pages per PDF")
    
    args = parser.parse_args()
    convert_pdfs(args.input_dir, args.output_dir, args.cuda, args.threads, args.max_pages)


if __name__ == "__main__":
    main()
