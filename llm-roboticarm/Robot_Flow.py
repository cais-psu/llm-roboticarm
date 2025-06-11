
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="SV6Y70SgNrmvw9mLNlcS"
)

result = client.run_workflow(
    workspace_name="gearassemblydetection",
    workflow_id="detect-count-and-visualize-2",
    images={
        "image": "vision_data/IMG_4809.jpg"
    },
    use_cache=True # cache workflow definition for 15 minutes
)
