from huggingface_hub import InferenceClient

client = None

def get_client(api_key : str):
    global client
    if client is None:
        client = InferenceClient(
            provider="nscale",
            api_key=api_key,
        )

    return client


def run_model(model_id : str, api_key : str, test_message):
    _client = get_client(api_key)
    print(api_key)
    print(model_id)
    print(test_message)
    completion = _client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": test_message
            }
        ],
    )

    return completion.choices[0].message
