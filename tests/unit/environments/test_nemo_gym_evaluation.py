# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw evaluation preserves verifier results independently of training capture."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nemo_rl.environments.nemo_gym import NemoGym


def test_raw_evaluation_keeps_tokenless_success_and_infrastructure_sidecar():
    rows = [{"_rowidx": 0}, {"_rowidx": 1}]
    results = [
        {"reward": 1, "evaluation_completed": True, "response": {"output": []}},
        {"_ng_failure_response_body": {"status": 503}},
    ]

    async def pending(row, result):
        return row, result

    run_examples = Mock(
        side_effect=lambda **kwargs: [
            pending(row, result) for row, result in zip(kwargs["examples"], results)
        ]
    )
    actor = SimpleNamespace(
        _require_spinup=Mock(),
        _forest_training=False,
        _token_capture_enabled=False,
        rch=SimpleNamespace(run_examples=run_examples),
        head_server_config={},
    )

    async def collect():
        return [
            record
            async for record in NemoGym.__ray_metadata__.modified_class.run_evaluation(
                actor, rows
            )
        ]

    recorded = asyncio.run(collect())
    assert [record["result"] for record in recorded] == results
    assert [record["row_index"] for record in recorded] == [0, 1]
    assert run_examples.call_args.kwargs["route_failures_to_sidecar"] is True


def test_raw_evaluation_preserves_yielded_rows_when_later_transport_fails():
    async def success():
        return {"_rowidx": 0}, {"evaluation_completed": True, "reward": 0}

    async def failure():
        raise ConnectionError("Gym connection lost")

    actor = SimpleNamespace(
        _require_spinup=Mock(),
        _forest_training=False,
        _token_capture_enabled=False,
        rch=SimpleNamespace(run_examples=lambda **kwargs: [success(), failure()]),
        head_server_config={},
    )
    records = []

    async def collect():
        async for record in NemoGym.__ray_metadata__.modified_class.run_evaluation(
            actor, [{"_rowidx": 0}, {"_rowidx": 1}]
        ):
            records.append(record)

    with pytest.raises(ConnectionError, match="connection lost"):
        asyncio.run(collect())
    assert len(records) == 1
    assert records[0]["result"]["reward"] == 0


@pytest.mark.parametrize("capture,forest", [(True, False), (True, True)])
def test_raw_evaluation_cannot_skip_capture_drain(capture, forest):
    actor = SimpleNamespace(
        _require_spinup=Mock(),
        _token_capture_enabled=capture,
        _forest_training=forest,
    )

    async def collect():
        return [
            record
            async for record in NemoGym.__ray_metadata__.modified_class.run_evaluation(
                actor, [{"_rowidx": 0}]
            )
        ]

    with pytest.raises(ValueError, match="non-capture Gym actor"):
        asyncio.run(collect())
