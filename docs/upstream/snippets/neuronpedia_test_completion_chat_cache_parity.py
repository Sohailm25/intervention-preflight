"""Example Neuronpedia cache parity test for chat steering.

Intended target:
- apps/inference/tests/integration/test_completion_chat.py
"""

from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.np_steer_chat_message import NPSteerChatMessage
from neuronpedia_inference_client.models.np_steer_method import NPSteerMethod
from neuronpedia_inference_client.models.np_steer_type import NPSteerType
from neuronpedia_inference_client.models.steer_completion_chat_post200_response import (
    SteerCompletionChatPost200Response,
)
from neuronpedia_inference_client.models.steer_completion_chat_post_request import (
    SteerCompletionChatPostRequest,
)

from tests.conftest import (
    FREQ_PENALTY,
    MODEL_ID,
    N_COMPLETION_TOKENS,
    SEED,
    STEER_SPECIAL_TOKENS,
    STRENGTH_MULTIPLIER,
    TEMPERATURE,
    TEST_PROMPT,
    X_SECRET_KEY,
)
from tests.integration.test_completion_chat import ENDPOINT, TEST_STEER_FEATURE
from tests.utils.assertions import assert_deterministic_output_match


def _build_request() -> SteerCompletionChatPostRequest:
    return SteerCompletionChatPostRequest(
        prompt=[NPSteerChatMessage(content=TEST_PROMPT, role="user")],
        model=MODEL_ID,
        steer_method=NPSteerMethod.SIMPLE_ADDITIVE,
        normalize_steering=False,
        types=[NPSteerType.STEERED, NPSteerType.DEFAULT],
        features=[TEST_STEER_FEATURE],
        n_completion_tokens=N_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        strength_multiplier=STRENGTH_MULTIPLIER,
        freq_penalty=FREQ_PENALTY,
        seed=SEED,
        steer_special_tokens=STEER_SPECIAL_TOKENS,
    )


def _run_request(
    client: TestClient,
    *,
    use_past_kv_cache: bool,
):
    # Wire this flag into the underlying generation path using either:
    # - a test fixture
    # - monkeypatch
    # - a temporary local parameter if the endpoint does not expose it yet
    request = _build_request()
    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert response.status_code == 200
    parsed = SteerCompletionChatPost200Response(**response.json())
    return {output.type: output.raw for output in parsed.outputs}


def test_completion_chat_cache_parity_features_additive(client: TestClient):
    cached = _run_request(client, use_past_kv_cache=True)
    uncached = _run_request(client, use_past_kv_cache=False)

    assert_deterministic_output_match(
        cached[NPSteerType.DEFAULT],
        uncached[NPSteerType.DEFAULT],
        left_label="cached default output",
        right_label="uncached default output",
    )
    assert_deterministic_output_match(
        cached[NPSteerType.STEERED],
        uncached[NPSteerType.STEERED],
        left_label="cached steered output",
        right_label="uncached steered output",
    )
