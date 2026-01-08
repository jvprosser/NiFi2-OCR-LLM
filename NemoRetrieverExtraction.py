from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import PropertyDescriptor, StandardValidators
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.primitives.jobs import JobSpec
from nv_ingest_client.client.interface import Ingestor
import json

class NemoRetrieverExtraction(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']

    class ProcessorDetails:
        version = '2.0.0'
        description = 'Extracts structured Markdown and metadata using NVIDIA NeMo Retriever.'
        tags = ['nvidia', 'nemo', 'ocr', 'ai', 'agentic']
        dependencies = ['nv-ingest-client', 'torch']

    def __init__(self, **kwargs):
        self.NIM_HOST = PropertyDescriptor(
            name="NIM Host",
            description="IP/Hostname of the NeMo Retriever NIM.",
            required=True,
            default_value="localhost",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR]
        )
        self.HANDWRITING_THRESHOLD = PropertyDescriptor(
            name="Handwriting Confidence Threshold",
            description="The score (0-1) below which we flag a document for 'Thinking' escalation.",
            required=True,
            default_value="0.75",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR]
        )
        self.descriptors = [self.NIM_HOST, self.HANDWRITING_THRESHOLD]

    def getPropertyDescriptors(self):
        return self.descriptors

    def transform(self, context, flowfile):
        host = context.getProperty(self.NIM_HOST).getValue()
        threshold = float(context.getProperty(self.HANDWRITING_THRESHOLD).getValue())
        content = flowfile.getContentsAsBytes()

        try:
            client = NvIngestClient(message_client_hostname=host)

            # Agentic Job Setup: Using nemoretriever_parse for high-precision
            ingestor = (
                Ingestor(client=client)
                .files([content])
                .extract(
                    extract_method="nemoretriever_parse",
                    extract_text=True,
                    extract_tables=True
                )
            )

            # Run extraction
            extraction_results = ingestor.ingest()

            # Agentic Logic: Scan metadata for handwriting flags
            # nemoretriever-parse returns classes like 'paragraph', 'header', 'handwritten'
            needs_thinking_escalation = False
            for page in extraction_results:
                for element in page.get('metadata', []):
                    # Check if the model classified any block as 'handwritten'
                    # or if the OCR confidence is low
                    if element.get('class') == 'handwritten' or element.get('confidence', 1.0) < threshold:
                        needs_thinking_escalation = True
                        break

            # Prepare Output
            output_content = json.dumps(extraction_results)
            attributes = {
                "nemo.extraction.success": "true",
                "nemo.needs.escalation": str(needs_thinking_escalation).lower(),
                "mime.type": "application/json"
            }

            return FlowFileTransformResult(
                relationship="success",
                contents=output_content,
                attributes=attributes
            )

        except Exception as e:
            self.logger.error(f"NeMo NIM Error: {str(e)}")
            return FlowFileTransformResult(relationship="failure")
