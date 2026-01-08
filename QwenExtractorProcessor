from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.properties import PropertyDescriptor, StandardValidators
import json
import requests

class ClouderaAIThinkingEscalation(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']

    class ProcessorDetails:
        version = '2.0.0'
        description = 'Escalation processor using Qwen3-VL-Thinking to resolve complex OCR/Handwriting issues.'
        tags = ['cloudera', 'cml', 'qwen', 'vision', 'ai', 'thinking']

    def __init__(self, **kwargs):
        self.CML_ENDPOINT = PropertyDescriptor(
            name="CML Model Endpoint",
            description="The URL for the Cloudera AI Model API.",
            required=True,
            validators=[StandardValidators.NON_EMPTY_VALIDATOR]
        )
        self.API_KEY = PropertyDescriptor(
            name="Access Key",
            description="The API Key/Token for the CML Model.",
            required=True,
            sensitive=True,
            validators=[StandardValidators.NON_EMPTY_VALIDATOR]
        )
        self.THINKING_THRESHOLD = PropertyDescriptor(
            name="Thinking Quality Threshold",
            description="Minimum score from the model (0-1) to consider the escalation successful.",
            required=True,
            default_value="0.85",
            validators=[StandardValidators.NON_EMPTY_VALIDATOR]
        )
        self.descriptors = [self.CML_ENDPOINT, self.API_KEY, self.THINKING_THRESHOLD]

    def getPropertyDescriptors(self):
        return self.descriptors

    def transform(self, context, flowfile):
        endpoint = context.getProperty(self.CML_ENDPOINT).getValue()
        api_key = context.getProperty(self.API_KEY).getValue()
        threshold = float(context.getProperty(self.THINKING_THRESHOLD).getValue())
        
        # Original document bytes (PDF/Image)
        content_bytes = flowfile.getContentsAsBytes()

        # Prepare the payload for Qwen3-VL
        # Note: Depending on your CML setup, you may need to base64 encode the content
        payload = {
            "request": {
                "prompt": "Carefully transcribe the handwriting in this document. Think step-by-step.",
                "model": "Qwen3-VL-235B-A22B-Thinking",
                "stream": False
            }
        }

        try:
            # API Call to Cloudera AI Model
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            # Extract thinking score and transcription
            # Logic assumes CML returns a confidence or 'reasons' field
            ai_output = result.get('response', {})
            thinking_score = ai_output.get('confidence_score', 0.0)
            transcription = ai_output.get('text', '')

            # Determine if we met the secondary threshold
            final_success = thinking_score >= threshold
            
            attributes = {
                "escalation.resolved": str(final_success).lower(),
                "escalation.score": str(thinking_score),
                "ai.model.used": "Qwen3-VL-Thinking",
                "mime.type": "application/json"
            }

            # If failed even with thinking model, route to a manual human queue
            target_relationship = "success" if final_success else "failure"

            return FlowFileTransformResult(
                relationship=target_relationship,
                contents=json.dumps(result),
                attributes=attributes
            )

        except Exception as e:
            self.logger.error(f"Cloudera AI Error: {str(e)}")
            return FlowFileTransformResult(relationship="failure")
