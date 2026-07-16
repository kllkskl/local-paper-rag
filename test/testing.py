from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from pathlib import Path

converter = PdfConverter(artifact_dict=create_model_dict())

pdf_path = "data/papers/framework_paper.pdf"
rendered = converter(pdf_path)
text, _, images = text_from_rendered(rendered)

with open("test/text.txt", "w", encoding="utf-8") as f:
    f.write(text)