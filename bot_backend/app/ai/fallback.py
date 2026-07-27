from app.ai.retry import retry_request


def fallback_request(
    models,
    messages,
    stream=False
):

    last_error = None

    for index, model in enumerate(models):

        print("=" * 50)
        print(f"Trying Model : {model.name}")

        try:

            response = retry_request(
                model=model.model,
                messages=messages,
                temperature=0.4,
                max_tokens=250,
                stream=stream
            )

            if stream:
                return response

            return {
                "response": response,
                "provider": model.name,
                "model": model.model,
                "fallback": index > 0
            }

        except Exception as e:

            print(f"Failed : {model.name}")
            print(e)

            last_error = e

    raise Exception(last_error)