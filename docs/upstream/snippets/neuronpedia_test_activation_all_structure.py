"""Example Neuronpedia activation-all structure test.

Intended target:
- apps/inference/tests/integration/test_activation_all.py
"""

from fastapi.testclient import TestClient
from neuronpedia_inference_client.models.activation_all_post200_response import (
    ActivationAllPost200Response,
)
from neuronpedia_inference_client.models.activation_all_post_request import (
    ActivationAllPostRequest,
)

from tests.conftest import (
    BOS_TOKEN_STR,
    MODEL_ID,
    SAE_SELECTED_SOURCES,
    SAE_SOURCE_SET,
    TEST_PROMPT,
    X_SECRET_KEY,
)
from tests.utils.assertions import assert_activation_structure_stable

ENDPOINT = "/v1/activation/all"


def _activation_rows(response_model: ActivationAllPost200Response) -> list[list[float]]:
    return [list(activation.values) for activation in response_model.activations]


def test_activation_all_structure(client: TestClient):
    request = ActivationAllPostRequest(
        prompt=TEST_PROMPT,
        model=MODEL_ID,
        source_set=SAE_SOURCE_SET,
        selected_sources=SAE_SELECTED_SOURCES,
        sort_by_token_indexes=[],
        num_results=5,
        ignore_bos=True,
    )

    response = client.post(
        ENDPOINT,
        json=request.model_dump(),
        headers={"X-SECRET-KEY": X_SECRET_KEY},
    )
    assert response.status_code == 200

    parsed = ActivationAllPost200Response(**response.json())
    rows = _activation_rows(parsed)

    assert len(parsed.activations) == 5
    assert parsed.tokens == [BOS_TOKEN_STR, "Hello", ",", " world", "!"]
    assert all(activation.source in SAE_SELECTED_SOURCES for activation in parsed.activations)
    assert all(len(row) == len(parsed.tokens) for row in rows)

    # Replace this reference snapshot with either:
    # - a regenerated structural snapshot checked into the repo
    # - a small fixture created during the test setup
    reference_rows = rows

    assert_activation_structure_stable(
        rows,
        reference_rows,
        min_mean_cosine=0.90,
        min_mean_topk_overlap=0.50,
        top_k=3,
    )
