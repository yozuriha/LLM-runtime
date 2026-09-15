from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic


class RequestState(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class GenerationRequest:
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
        return len(self.input_ids)

    @property
    def seq_len(self) -> int:
        return self.prompt_len + len(self.generated_ids)

    @property
    def finished(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.CANCELLED)

    def append_token(self, token_id: int) -> None:
        self.generated_ids.append(int(token_id))
        self.next_token = int(token_id)
        if len(self.generated_ids) >= self.max_new_tokens or token_id == self.eos_token_id:
            self.state = RequestState.FINISHED
            self.finished_at = monotonic()
