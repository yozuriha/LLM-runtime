import torch

from llm_runtime import ContinuousKVCache, GenerationEngine, GenerationRequest
from llm_runtime.runner import ModelRunner
from llm_runtime.scheduler import FIFOScheduler


class CountingRunner(ModelRunner):
    def prefill(self, request):
        return torch.tensor([0.0, 10.0])

    def decode(self, request, token_id):
        return torch.tensor([10.0, 0.0])


def test_cache_accounting():
    cache = ContinuousKVCache(10)
    cache.allocate("a", 3, "past")
    cache.update("a", 5)
    assert cache.used_tokens == 5
    cache.release("a")
    assert cache.used_tokens == 0


def test_engine_generates_and_releases_cache():
    first = GenerationRequest("a", [4, 5], max_new_tokens=2)
    second = GenerationRequest("b", [7], max_new_tokens=1)
    engine = GenerationEngine(CountingRunner(), max_batch_size=2)
    engine.add_request(first)
    engine.add_request(second)
    completed = engine.run()
    assert {r.request_id for r in completed} == {"a", "b"}
    assert first.generated_ids == [1, 0]
    assert second.generated_ids == [1]
    assert engine.kv_cache.used_tokens == 0


def test_scheduler_respects_batch_and_token_limits():
    scheduler = FIFOScheduler(max_batch_size=2, max_num_batched_tokens=4)
    requests = [GenerationRequest(str(i), [i] * length) for i, length in enumerate((2, 2, 1))]
    for request in requests:
        scheduler.add_request(request)
    assert [r.request_id for r in scheduler.next_batch()] == ["0", "1"]
    scheduler.finish(requests[0])
    assert [r.request_id for r in scheduler.next_batch()] == ["1", "2"]
