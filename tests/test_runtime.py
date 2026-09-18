import torch
from types import SimpleNamespace

from llm_runtime import ContinuousKVCache, GenerationEngine, GenerationRequest, PagedKVCache
from llm_runtime.runner import ModelRunner
from llm_runtime.scheduler import FIFOScheduler
from llm_runtime.runner import PyTorchCausalLMRunner


class CountingRunner(ModelRunner):
    def __init__(self):
        self.prefill_batch_calls = 0
        self.decode_batch_calls = 0

    def prefill(self, request):
        return torch.tensor([0.0, 10.0])

    def decode(self, request, token_id):
        return torch.tensor([10.0, 0.0])

    def prefill_batch(self, requests):
        self.prefill_batch_calls += 1
        return super().prefill_batch(requests)

    def decode_batch(self, requests, token_ids):
        self.decode_batch_calls += 1
        return super().decode_batch(requests, token_ids)


class FakeCausalLM(torch.nn.Module):
    """Small CPU model exposing tuple KV tensors for runner shape tests."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, past_key_values=None, **kwargs):
        current = input_ids.float().unsqueeze(1).unsqueeze(-1)
        if past_key_values is None:
            key = value = current
        else:
            old_key, old_value = past_key_values[0]
            key = torch.cat((old_key, current), dim=2)
            value = torch.cat((old_value, current), dim=2)
        logits = torch.zeros((*input_ids.shape, 4))
        logits.scatter_(2, (input_ids.remainder(4)).unsqueeze(-1), 1.0)
        return SimpleNamespace(logits=logits, past_key_values=((key, value),))


class FakeLayerCache:
    def __init__(self, key, value):
        self.keys = key
        self.values = value


class FakeNewCache:
    """Transformers 5-style cache with layers instead of to_legacy_cache."""

    def __init__(self, pairs):
        self.layers = [FakeLayerCache(key, value) for key, value in pairs]


def test_cache_accounting():
    cache = ContinuousKVCache(10)
    cache.allocate("a", 3, "past")
    cache.update("a", 5)
    assert cache.used_tokens == 5
    cache.release("a")
    assert cache.used_tokens == 0


def test_paged_kv_cache_allocates_blocks_and_materializes_pages():
    cache = PagedKVCache(max_tokens=8, block_size=2)
    key = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5, 1)
    value = key + 10
    cache.allocate("a", 5, ((key, value),))
    assert cache.block_table("a") == (0, 1, 2)
    assert cache.used_blocks == 3
    materialized = cache.get("a")
    torch.testing.assert_close(materialized[0][0], key)
    cache.update("a", 6, ((torch.cat((key, key[:, :, :1]), dim=2),
                            torch.cat((value, value[:, :, :1]), dim=2)),))
    assert cache.used_tokens == 6
    assert cache.used_blocks == 3
    cache.release("a")
    assert cache.used_tokens == 0
    assert cache.free_blocks == cache.max_blocks


def test_engine_generates_and_releases_cache():
    first = GenerationRequest("a", [4, 5], max_new_tokens=2)
    second = GenerationRequest("b", [7], max_new_tokens=1)
    runner = CountingRunner()
    engine = GenerationEngine(runner, max_batch_size=2)
    engine.add_request(first)
    engine.add_request(second)
    completed = engine.run()
    assert {r.request_id for r in completed} == {"a", "b"}
    assert first.generated_ids == [1, 0]
    assert second.generated_ids == [1]
    assert engine.kv_cache.used_tokens == 0
    assert runner.prefill_batch_calls == 1
    assert runner.decode_batch_calls == 1


def test_pytorch_runner_batches_variable_length_kv():
    runner = PyTorchCausalLMRunner(FakeCausalLM(), device="cpu")
    requests = [GenerationRequest("long", [1, 2]), GenerationRequest("short", [3])]

    logits = runner.prefill_batch(requests)
    assert logits.shape == (2, 4)
    assert [request.past_key_values[0][0].shape[2] for request in requests] == [2, 1]

    decode_logits = runner.decode_batch(requests, [0, 1])
    assert decode_logits.shape == (2, 4)
    assert [request.past_key_values[0][0].shape[2] for request in requests] == [3, 2]
    assert requests[0].past_key_values[0][0].flatten().tolist() == [1.0, 2.0, 0.0]
    assert requests[1].past_key_values[0][0].flatten().tolist() == [3.0, 1.0]


def test_runner_accepts_transformers_new_cache_layout():
    key = torch.randn(2, 1, 3, 4)
    value = torch.randn(2, 1, 3, 4)
    requests = [GenerationRequest("a", [1, 2]), GenerationRequest("b", [3])]
    requests[0].past_key_values = FakeNewCache(((key[:1], value[:1]),))
    requests[1].past_key_values = FakeNewCache(((key[1:], value[1:]),))

    stacked, max_len, lengths = PyTorchCausalLMRunner._stack_past(requests)
    stacked_pairs = PyTorchCausalLMRunner._legacy_past(stacked)
    assert stacked_pairs[0][0].shape == (2, 1, 3, 4)
    assert max_len == 3
    assert lengths == [3, 3]


def test_scheduler_respects_batch_and_token_limits():
    scheduler = FIFOScheduler(max_batch_size=2, max_num_batched_tokens=4)
    requests = [GenerationRequest(str(i), [i] * length) for i, length in enumerate((2, 2, 1))]
    for request in requests:
        scheduler.add_request(request)
    assert [r.request_id for r in scheduler.next_batch()] == ["0", "1"]
    scheduler.finish(requests[0])
    assert [r.request_id for r in scheduler.next_batch()] == ["1", "2"]
