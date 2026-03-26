from huggingface_hub import HfApi

api = HfApi()
info = api.model_info("mistralai/Mistral-7B-Instruct-v0.3", expand="inferenceProviderMapping")

if info.inference_provider_mapping:
    print("Supported providers for mistralai/Mistral-7B-Instruct-v0.3:")
    for provider in info.inference_provider_mapping:
        print(f"\nProvider: {provider.provider}")
        print(f"  Task: {provider.task}")
        print(f"  Status: {provider.status}")