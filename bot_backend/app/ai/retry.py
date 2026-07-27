from litellm import completion
import time


def retry_request(
    model,
    messages,
    temperature=0.4,
    max_tokens=250,
    retries=3,
    stream=False
):

    for attempt in range(retries):

        try:

            print(f"Retry {attempt + 1} using {model}")

            return completion(

                model=model,

                messages=messages,

                temperature=temperature,

                max_tokens=max_tokens,

                stream=stream

            )

        except Exception as e:

            print(f"Retry {attempt + 1} Failed")
            print(e)

            time.sleep(1)

    raise Exception("All Retry Attempts Failed")