import enum
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Mapping,
                    Optional, Union)

import torch

from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest

if TYPE_CHECKING:
    from vllm.inputs import LLMInputs
    from vllm.multimodal.base import MultiModalDataDict


class Request:

    def __init__(
        self,
        request_id: str,
        inputs: "LLMInputs",
        arrival_time: float,
        sampling_params: SamplingParams,
        lora_request: Optional[LoRARequest] = None,
    ) -> None:
        self.request_id = request_id
        self.inputs = inputs
        self.metrics = RequestMetrics(arrival_time=arrival_time,
                                      last_token_time=arrival_time,
                                      first_scheduled_time=None,
                                      first_token_time=None,
                                      time_in_queue=None)
        self.sampling_params = sampling_params
        self.lora_request = lora_request

        self.status = RequestStatus.WAITING
        self.stop_reason: Union[int, str, None] = None
        self.num_prompt_tokens = len(inputs["prompt_token_ids"])
        self.num_computed_tokens = 0

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_num_prefill_tokens(self) -> int:
        return self.num_prompt_tokens - self.num_computed_tokens

    def has_prefill(self) -> bool:
        return self.num_prompt_tokens > self.num_computed_tokens


class RequestStatus(enum.IntEnum):
    """Status of a sequence."""
    WAITING = 0
    RUNNING = 1
    SWAPPED = 2
    # Note: anything after SWAPPED (2) will be considered
    # as a finished status.
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.SWAPPED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> Union[str, None]:
        if status == RequestStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == RequestStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == RequestStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == RequestStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


@dataclass
class RequestMetrics:
    """Metrics associated with a request.

    Attributes:
        arrival_time: The time when the request arrived.
        first_scheduled_time: The time when the request was first scheduled.
        first_token_time: The time when the first token was generated.
        time_in_queue: The time the request spent in the queue.
        finished_time: The time when the request was finished.
        scheduler_time: The time spent in the scheduler when this request was
                        being considered by the scheduler.
        model_forward_time: The time spent in the model forward pass when this
                            request was in the batch.
        model_execute_time: The time spent in the model execute function. This
                            will include model forward, block/sync across
                            workers, cpu-gpu sync time and sampling time.
    """
    arrival_time: float
    last_token_time: float
    first_scheduled_time: Optional[float]
    first_token_time: Optional[float]
    time_in_queue: Optional[float]
    finished_time: Optional[float] = None
    scheduler_time: Optional[float] = None
    model_forward_time: Optional[float] = None
    model_execute_time: Optional[float] = None
