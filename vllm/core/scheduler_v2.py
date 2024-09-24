import enum
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (Callable, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)

from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.core.interfaces import AllocStatus, BlockSpaceManager
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.utils import Device

from vllm.request import Request, RequestStatus
from vllm.sampling_params import SamplingParams
from vllm.multimodal import MultiModalDataDict

logger = init_logger(__name__)


@dataclass
class NewRequestData:

    req_id: str
    prompt_token_ids: List[int]
    prompt: Optional[str]
    multi_modal_data: Optional[MultiModalDataDict]
    sampling_params: SamplingParams
    block_ids: List[int]
    num_scheduled_tokens: int


@dataclass
class InFlightRequestData:

    req_id: str
    new_block_ids: List[int]
    num_scheduled_tokens: int


@dataclass
class SchedulerOutput:

    num_scheduled_reqs: int
    num_scheduled_new_reqs: int
    num_scheduled_in_flight_reqs: int

    scheduled_new_reqs: List[NewRequestData]
    scheduled_in_flight_reqs: List[InFlightRequestData]
    num_scheduled_tokens: Dict[str, int]
    total_num_scheduled_tokens: int

    stopped_req_ids: List[str]
    ignored_req_ids: List[str]
    aborted_req_ids: List[str]
    preempted_req_ids: List[str]

    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_swap_out: List[Tuple[int, int]]


@dataclass
class SchedulingBudget:

    max_num_running_reqs: int
    curr_num_running_reqs: int

    max_num_scheduled_tokens: int
    curr_num_scheduled_tokens: int

    @staticmethod
    def _get_num_tokens(request: Request, num_tokens: int) -> int:
        assert num_tokens >= 0
        if num_tokens == 0:
            if request.has_prefill():
                num_tokens = request.get_num_prefill_tokens()
            else:
                num_tokens = 1
        else:
            if request.has_prefill():
                assert num_tokens <= request.get_num_prefill_tokens()
            else:
                assert num_tokens == 1
        return num_tokens

    def schedule_request(self, request: Request, num_tokens: int = 0) -> None:
        num_tokens = self._get_num_tokens(request, num_tokens)
        self.curr_num_running_reqs += 1
        self.curr_num_scheduled_tokens += num_tokens

    def can_schedule(self, request: Request, num_tokens: int = 0) -> bool:
        if self.curr_num_running_reqs + 1 > self.max_num_running_reqs:
            return False
        num_tokens = self._get_num_tokens(request, num_tokens)
        if self.curr_num_scheduled_tokens + num_tokens > self.max_num_scheduled_tokens:
            return False


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        # Note for LoRA scheduling: the current policy is extremely
        # simple and NOT fair. It can lead to starvation of some
        # LoRAs. This should be improved in the future.
        self.lora_config = lora_config

        version = "v1"
        if self.scheduler_config.use_v2_block_manager:
            version = "v2"
        if self.scheduler_config.embedding_mode:
            version = "embedding"
        BlockSpaceManagerImpl = BlockSpaceManager.get_block_space_manager_class(
            version)
        num_gpu_blocks = cache_config.num_gpu_blocks
        num_cpu_blocks = cache_config.num_cpu_blocks
        # Create the block space manager.
        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        self.waiting: Deque[Request] = deque()
        self.preempted: Deque[Request] = deque()
        self.running: Deque[Request] = deque()
        self.swapped: Deque[Request] = deque()

    def schedule(self) -> SchedulerOutput:
        # TODO(woosuk): Implement chunked prefills.
        budget = SchedulingBudget(
            max_num_scheduled_reqs=self.scheduler_config.max_num_seqs,
            curr_num_scheduled_reqs=len(self.running),
            max_num_scheduled_tokens=self.scheduler_config.max_num_batched_tokens,
            curr_num_scheduled_tokens=0,
        )

        if self.swapped:
            ...

        while self.swapped:
            req = self.swapped[0]
            if budget.can_schedule(req):
                self.swapped.popleft()
                budget.schedule_request(req)
                self.running.append(req)

        if self.swapped:
            self.swapped.append()
            pass

        if not self.swapped:
            prefills = ...

        if not prefills:
            decode = ...




        for seq_group in self.running:
            budget.add_num_seqs(seq_group.request_id,
                                seq_group.get_max_num_running_seqs())
        curr_loras = set(
            seq_group.lora_int_id for seq_group in self.running
            if seq_group.lora_int_id > 0) if self.lora_enabled else None

        # If any requests are swapped, prioritized swapped requests.
        if not self.swapped:
            prefills = self._schedule_prefills(budget,
                                               curr_loras,
                                               enable_chunking=False)

        # Don't schedule decodes if prefills are scheduled.
        # NOTE: If `_schedule_prefills` doesn't enable chunking, self.running
        # only contains decode requests, not chunked prefills.
        if len(prefills.seq_groups) == 0:
            running_scheduled = self._schedule_running(budget,
                                                       curr_loras,
                                                       enable_chunking=False)

            # If any sequence group is preempted, do not swap in any sequence
            # group. because it means there's no slot for new running requests.
            if len(running_scheduled.preempted) + len(
                    running_scheduled.swapped_out) == 0:
                swapped_in = self._schedule_swapped(budget, curr_loras)

        assert (budget.num_batched_tokens <=
                self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_seqs

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)
        # Update new running requests.
        if len(prefills.seq_groups) > 0:
            self.running.extend([s.seq_group for s in prefills.seq_groups])

        self.running.extend(running_scheduled.decode_seq_groups_list)

        if len(swapped_in.decode_seq_groups) > 0:
            self.running.extend(
                [s.seq_group for s in swapped_in.decode_seq_groups])

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)
        preempted = (len(running_scheduled.preempted) +
                     len(running_scheduled.swapped_out))

        # There should be no prefill from running queue because this policy
        # doesn't allow chunked prefills.
        assert len(running_scheduled.prefill_seq_groups) == 0
        assert len(swapped_in.prefill_seq_groups) == 0

        # Merge lists
        num_prefill_groups = len(prefills.seq_groups)
        if num_prefill_groups > 0:
            scheduled_seq_groups = prefills.seq_groups
            scheduled_seq_groups.extend(running_scheduled.decode_seq_groups)
        else:
            scheduled_seq_groups = running_scheduled.decode_seq_groups
        scheduled_seq_groups.extend(swapped_in.decode_seq_groups)

        blocks_to_copy = running_scheduled.blocks_to_copy
        blocks_to_copy.extend(swapped_in.blocks_to_copy)

        ignored_seq_groups = prefills.ignored_seq_groups
        ignored_seq_groups.extend(swapped_in.infeasible_seq_groups)

        return SchedulerOutputs(
            scheduled_seq_groups=scheduled_seq_groups,
            num_prefill_groups=num_prefill_groups,
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=running_scheduled.num_lookahead_slots,
            running_queue_size=len(self.running),
            preempted=preempted,
        )


    def _schedule_running(
        self,
        budget: "SchedulingBudget",
        curr_loras: Optional[Set[int]],
    ):
        blocks_to_swap_out: List[Tuple[int, int]] = []
        swapped_reqs: List[Request] = []

        new_running: Deque[Request] = deque()
        while self.running:
            req = self.running[0]

            # FIXME
            num_tokens = 1
            assert not budget.can_schedule(req, num_tokens)

            while not self.

        running_queue = self.running
        while running_queue:
            seq_group = running_queue[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()

            # NOTE(woosuk): Preemption happens only when there is no available
            # slot to keep all the sequence groups in the RUNNING state.
            while not self._can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if (curr_loras is not None and seq_group.lora_int_id > 0
                        and seq_group.lora_int_id in curr_loras):
                    curr_loras.remove(seq_group.lora_int_id)

                # Determine victim sequence
                cont_loop = True
                if running_queue:
                    # Preempt the lowest-priority sequence group.
                    victim_seq_group = running_queue.pop()
                else:
                    # No other sequence group can be preempted.
                    # Preempt the current sequence group.
                    # Note: This is also where we stop this loop
                    # (since there is nothing else to preempt)
                    victim_seq_group = seq_group
                    cont_loop = False

                # With async postprocessor, before preempting a sequence
                # we need to ensure it has no pending async postprocessor
                do_preempt = True

                # Do preemption
                if do_preempt:
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_seq_group)
                    else:
                        swapped_out.append(victim_seq_group)

                if not cont_loop:
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy)
                is_prefill = seq_group.is_prefill()

                scheduled_seq_group: ScheduledSequenceGroup = \
                    self._scheduled_seq_group_cache[self.cache_id].get_object()
                scheduled_seq_group.seq_group = seq_group
                if is_prefill:
                    scheduled_seq_group.token_chunk_size = num_running_tokens
                    prefill_seq_groups.append(scheduled_seq_group)
                    ret.prefill_seq_groups_list.append(seq_group)
                else:
                    scheduled_seq_group.token_chunk_size = 1
                    decode_seq_groups.append(scheduled_seq_group)
                    ret.decode_seq_groups_list.append(seq_group)

                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)

        return ret


    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def abort_requests(self, request_ids: Union[str, Iterable[str]]) -> None:
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        request_ids = set(request_ids)

        queues = [self.waiting, self.preempted, self.running, self.swapped]
        for queue in queues:
            aborted_reqs: List[Request] = []
            for request in queue:
                if not request_ids:
                    break
                if request.request_id in request_ids:
                    request.status = RequestStatus.FINISHED_ABORTED
                    aborted_reqs.append(request)
                    request_ids.remove(request.request_id)
            for request in aborted_reqs:
                queue.remove(request)

    def has_unfinished_requests(self) -> bool:
        return self.waiting or self.preempted or self.running or self.swapped

    def get_num_unfinished_requests(self) -> int:
        return (len(self.waiting) + len(self.preempted) + len(
            self.running) + len(self.swapped))

    def free_finished_requests(self) -> None:
        remaining: Deque[Request] = deque()
        for request in self.running:
            if request.is_finished():
                self.block_manager.free(request)
            else:
                remaining.append(request)
        self.running = remaining

    def _get_prompt_limit(self, request: Request) -> int:
        if self.scheduler_config.chunked_prefill_enabled:
            prompt_limit = self.scheduler_config.max_model_len
        else:
            prompt_limit = min(self.scheduler_config.max_model_len,
                               self.scheduler_config.max_num_batched_tokens)

        # Model is fine tuned with long context. Return the fine tuned max_len.
        if request.lora_request and request.lora_request.long_lora_max_len:
            assert prompt_limit <= request.lora_request.long_lora_max_len
            return request.lora_request.long_lora_max_len
        else:
            return prompt_limit

    @property
    def lora_enabled(self) -> bool:
        return bool(self.lora_config)


class PreemptionMode(enum.Enum):
    """Preemption modes.

    1. Swapping: Swap out the blocks of the preempted sequences to CPU memory
    and swap them back in when the sequences are resumed.
    2. Recomputation: Discard the blocks of the preempted sequences and
    recompute them when the sequences are resumed, treating the sequences as
    new prompts.
    """
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()
