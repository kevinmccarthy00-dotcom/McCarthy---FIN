"""Pre-flight check: one tiny request to confirm the API key, network, and model.

Costs a handful of tokens. Writes nothing to disk and never prints the key.

Usage: python scripts/test_api_connection.py
"""

import os
import sys

from run_evaluation import MODEL


def main():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit("FAIL: ANTHROPIC_API_KEY is not set in this environment.")
    print("ANTHROPIC_API_KEY is set (value hidden).")

    try:
        import anthropic
    except ImportError:
        sys.exit("FAIL: the 'anthropic' package is not installed (pip install -r requirements.txt).")

    client = anthropic.Anthropic(max_retries=1, timeout=30.0)
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=10,
            extra_body={"temperature": 1.0},
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
        )
    except anthropic.AuthenticationError:
        sys.exit("FAIL: authentication error (401) - the key is invalid or revoked.")
    except anthropic.PermissionDeniedError:
        sys.exit("FAIL: permission denied (403) - the key cannot use this model, or the network blocked it.")
    except anthropic.NotFoundError:
        sys.exit(f"FAIL: model '{MODEL}' not found (404).")
    except anthropic.RateLimitError:
        sys.exit("FAIL: rate limited (429) - check the account's credit balance / limits.")
    except anthropic.APIConnectionError as e:
        sys.exit(f"FAIL: could not reach the API ({type(e).__name__}). Check network access.")
    except anthropic.APIError as e:
        sys.exit(f"FAIL: API error {getattr(e, 'status_code', '')} ({type(e).__name__}).")

    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    print(f"PASS: model={resp.model} reply={text!r} "
          f"tokens in/out={resp.usage.input_tokens}/{resp.usage.output_tokens}")


if __name__ == "__main__":
    main()
