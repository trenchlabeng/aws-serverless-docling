# docling_parser.py (Refactored for VLM Pipeline)

import logging
import textwrap
from io import BytesIO

# --- Docling Imports for VLM Pipeline ---
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions,ResponseFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling_core.types.doc import DoclingDocument
from docling.datamodel.base_models import InputFormat
from docling.backend.msword_backend import MsWordDocumentBackend

logger = logging.getLogger(__name__)

class VlmDocParser:
    """
    A refactored parser dedicated to converting documents using a VLM pipeline
    via a LiteLLM proxy.
    """
    def __init__(self, bytes_content: bytes, vlm_proxy_url: str):
        """
        Initializes the parser with the document content and the URL to the LiteLLM proxy.
        """
        if not bytes_content:
            raise ValueError("bytes_content must be provided")
        self.bytes_stream = bytes_content
        self.vlm_proxy_url = vlm_proxy_url

    def _configure_converter(self) -> DocumentConverter:
        """
        Configures the DocumentConverter to use the VlmPipeline, pointing to the
        EC2 proxy instance. This replaces the old, complex configuration.
        """
        # This prompt is sent to the VLM for each page of the document
        reconstruction_prompt = textwrap.dedent("""
            Analyze the provided image and raw text of a document page.
            Your task is to return only the clean, plain text representation of this page as if you were reading it naturally.
            Do not add any commentary or explanations. Only return the page's content.
            RAW_TEXT_START
            #RAW_TEXT#
            RAW_TEXT_END
        """)

        # Configure the VLM pipeline options
        pipeline_options = VlmPipelineOptions(
            enable_remote_services=True, # Allows docling to read from memory streams
            generate_page_images=True,    # CRITICAL: Tells docling to extract images for the VLM
        )

        # Configure the API endpoint for the VLM pipeline
        pipeline_options.vlm_options = ApiVlmOptions(
            url=f"{self.vlm_proxy_url}/chat/completions",
            params={"model": "us.amazon.nova-2-lite-v1:0"}, # This must match the model_name in your EC2's config.yaml
            prompt=reconstruction_prompt,
            response_format=ResponseFormat.MARKDOWN
        )

        # Create the converter, telling it to use our VLM pipeline for PDFs and images
        converter = DocumentConverter(
             allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.CSV,
                InputFormat.MD,
                InputFormat.XLSX,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    pipeline_cls=VlmPipeline,
                    backend=DoclingParseV4DocumentBackend,
                ),
                InputFormat.IMAGE: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pipeline_options,
                    backend=DoclingParseV4DocumentBackend
                ),
                InputFormat.DOCX: WordFormatOption(
                    backend=MsWordDocumentBackend
                ),
            }
        )
        return converter

    def reconstruct_document(self) -> DoclingDocument:
        """
        Runs the VLM conversion pipeline and returns a high-quality DoclingDocument object.
        """
        try:
            # The document source is now a simple in-memory stream
            source = DocumentStream(name="source_document", stream=BytesIO(self.bytes_stream))
            converter = self._configure_converter()
            
            print("Starting VLM document reconstruction...")
            conversion_result = converter.convert(source)
            print("VLM reconstruction complete.")
            
            return conversion_result.document
            
        except Exception as e:
            logger.error(f"Error during VLM document reconstruction: {str(e)}")
            raise