from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic


class RequestState(str, Enum):
    """单个生成请求在 Runtime 中经历的生命周期状态。"""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class GenerationRequest:
    """保存一次文本生成请求的输入、输出和推理阶段状态。

    ``past_key_values`` 由 Runner 写入，Engine 只负责在请求生命周期内
    持有它；这样后续替换为 paged KV cache 时，调度接口无需改变。
    """

    request_id: str
    input_ids: list[int]
    max_new_tokens: int = 32
    eos_token_id: int | None = None
    state: RequestState = RequestState.WAITING
    generated_ids: list[int] = field(default_factory=list)
    kv_slot: int | None = None
    past_key_values: object | None = None
    prefilled: bool = False
    next_token: int | None = None
    created_at: float = field(default_factory=monotonic)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def prompt_len(self) -> int:
        """返回 prompt 的 token 数，不包含新生成的 token。"""
        return len(self.input_ids)

    @property
    def seq_len(self) -> int:
        """返回当前完整序列长度（prompt 加已生成内容）。"""
        return self.prompt_len + len(self.generated_ids)

    @property
    def finished(self) -> bool:
        """请求是否已经结束，包含主动取消的情况。"""
        return self.state in (RequestState.FINISHED, RequestState.CANCELLED)

    def append_token(self, token_id: int) -> None:
        """追加一个采样结果，并根据 EOS 或长度上限更新状态。"""
        self.generated_ids.append(int(token_id))
        self.next_token = int(token_id)
        if len(self.generated_ids) >= self.max_new_tokens or token_id == self.eos_token_id:
            self.state = RequestState.FINISHED
            self.finished_at = monotonic()
