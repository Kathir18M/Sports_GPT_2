from app.ai.fallback import fallback_request
from app.core.config import get_models


def generate_response(prompt, mode):

    models = get_models(mode)

    if models is None:
        return {
            "error": "Mode must be Fast or Pro"
        }

    try:

        response = fallback_request(
            models=models,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            "response": response["response"].choices[0].message.content,
            "provider": response["provider"],
            "model": response["model"],
            "fallback": response["fallback"]
        }

    except Exception as e:
        return {
            "error": str(e)
        }


def stream_response(prompt, mode):

    models = get_models(mode)

    if models is None:
        yield "Mode must be Fast or Pro"
        return

    response = fallback_request(
        models=models,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=True
    )

    for chunk in response:
        if (
            chunk.choices
            and
            chunk.choices[0].delta.content
        ):
            yield chunk.choices[0].delta.content