from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, TypeVar, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler, Sampler

from mmprism.config import RuntimeConfig

_INTEGER = re.compile(r"0|[1-9][0-9]*")
_LAUNCH_KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
_T = TypeVar("_T")


class DistributedRunError(RuntimeError):
    """Raised when a formal distributed launch violates its runtime contract."""


def _environment_integer(name: str, value: str) -> int:
    if _INTEGER.fullmatch(value) is None:
        raise DistributedRunError(f"{name} must be a canonical non-negative integer")
    return int(value)


def _resolve_device(
    runtime: RuntimeConfig,
    *,
    local_rank: int,
    world_size: int,
) -> torch.device:
    accelerator = runtime.accelerator.lower()
    if accelerator not in {"auto", "cpu", "cuda", "gpu"}:
        raise DistributedRunError("runtime.accelerator must be auto, cpu, cuda, or gpu")
    if isinstance(runtime.devices, str) and runtime.devices != "auto":
        raise DistributedRunError("runtime.devices must be auto or an explicit device list")

    distributed = world_size > 1
    if accelerator == "cpu":
        if isinstance(runtime.devices, tuple):
            raise DistributedRunError("CPU runs cannot select CUDA device indices")
        device = torch.device("cpu")
    else:
        cuda_requested = accelerator in {"cuda", "gpu"} or isinstance(runtime.devices, tuple)
        if not torch.cuda.is_available():
            if cuda_requested:
                raise DistributedRunError("CUDA was requested but is unavailable")
            device = torch.device("cpu")
        else:
            if isinstance(runtime.devices, tuple):
                if distributed and local_rank >= len(runtime.devices):
                    raise DistributedRunError(
                        "LOCAL_RANK exceeds the explicit runtime.devices mapping"
                    )
                if not distributed and len(runtime.devices) != 1:
                    raise DistributedRunError(
                        "single-process runs require exactly one explicit device"
                    )
                index = runtime.devices[local_rank if distributed else 0]
            else:
                index = local_rank if distributed else 0
            if index >= torch.cuda.device_count():
                raise DistributedRunError(f"CUDA device index {index} is unavailable")
            device = torch.device("cuda", index)

    if device.type == "cpu" and runtime.precision != "32-true":
        raise DistributedRunError("CPU runs require runtime.precision=32-true")
    if (
        device.type == "cuda"
        and runtime.precision == "bf16-mixed"
        and not torch.cuda.is_bf16_supported()
    ):
        raise DistributedRunError("the selected CUDA device does not support bfloat16")
    return device


@dataclass(slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str | None
    _owns_process_group: bool = False

    @classmethod
    def from_environment(
        cls,
        runtime: RuntimeConfig,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> DistributedContext:
        values = os.environ if environment is None else environment
        present = {name: name in values for name in _LAUNCH_KEYS}
        if any(present.values()) and not all(present.values()):
            missing = ", ".join(name for name in _LAUNCH_KEYS if not present[name])
            raise DistributedRunError(
                f"distributed launch environment is incomplete; missing {missing}"
            )
        if all(present.values()):
            rank = _environment_integer("RANK", values["RANK"])
            local_rank = _environment_integer("LOCAL_RANK", values["LOCAL_RANK"])
            world_size = _environment_integer("WORLD_SIZE", values["WORLD_SIZE"])
            if world_size < 1:
                raise DistributedRunError("WORLD_SIZE must be positive")
            if rank >= world_size:
                raise DistributedRunError("RANK must be less than WORLD_SIZE")
            if world_size == 1 and (rank != 0 or local_rank != 0):
                raise DistributedRunError(
                    "a world-size-one launch requires RANK=LOCAL_RANK=0"
                )
        else:
            rank = 0
            local_rank = 0
            world_size = 1

        device = _resolve_device(runtime, local_rank=local_rank, world_size=world_size)
        backend = ("nccl" if device.type == "cuda" else "gloo") if world_size > 1 else None
        if world_size > 1:
            if not values.get("MASTER_ADDR"):
                raise DistributedRunError("distributed launch requires MASTER_ADDR")
            master_port = values.get("MASTER_PORT", "")
            if _INTEGER.fullmatch(master_port) is None or not 1 <= int(master_port) <= 65535:
                raise DistributedRunError(
                    "distributed launch requires MASTER_PORT within [1,65535]"
                )
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            backend=backend,
        )

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def initialize(self) -> None:
        if not self.enabled:
            return
        if not dist.is_available():
            raise DistributedRunError("torch.distributed is unavailable")
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        if dist.is_initialized():
            if (
                dist.get_rank() != self.rank
                or dist.get_world_size() != self.world_size
                or dist.get_backend() != self.backend
            ):
                raise DistributedRunError(
                    "existing process group does not match the launch environment"
                )
            return
        assert self.backend is not None
        try:
            dist.init_process_group(
                backend=self.backend,
                init_method="env://",
                rank=self.rank,
                world_size=self.world_size,
                timeout=timedelta(minutes=30),
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise DistributedRunError(f"unable to initialize process group: {error}") from error
        self._owns_process_group = True

    def close(self) -> None:
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        self._owns_process_group = False

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def rank_payload(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "device": str(self.device),
        }

    def topology_payload(self) -> dict[str, object]:
        ranks = self.all_gather_object(self.rank_payload())
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "world_size": self.world_size,
            "ranks": ranks,
        }

    def rank_zero_call(self, operation: Callable[[], _T], *, stage: str) -> _T:
        if not self.enabled:
            return operation()
        packets: list[object] = [None]
        if self.is_rank_zero:
            try:
                packets[0] = {"ok": True, "value": operation()}
            except Exception as error:  # noqa: BLE001 - propagate the same failure to every rank
                packets[0] = {
                    "ok": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
        if self.device.type == "cuda":
            dist.broadcast_object_list(packets, src=0, device=self.device)
        else:
            dist.broadcast_object_list(packets, src=0)
        packet = packets[0]
        if not isinstance(packet, Mapping) or not isinstance(packet.get("ok"), bool):
            raise DistributedRunError(f"rank-zero {stage} returned an invalid result")
        if packet["ok"] is not True:
            raise DistributedRunError(
                f"rank-zero {stage} failed: {packet.get('error_type')}: {packet.get('error')}"
            )
        return cast(_T, packet.get("value"))

    def all_gather_object(self, value: _T) -> list[_T]:
        if not self.enabled:
            return [value]
        gathered: list[object] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return cast(list[_T], gathered)

    def sum_values(self, values: Sequence[float | int]) -> tuple[float, ...]:
        tensor = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tuple(float(value) for value in tensor.cpu().tolist())

    def max_value(self, value: float) -> float:
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.cpu().item())

    def wrap_model(self, model: nn.Module) -> nn.Module:
        if not self.enabled:
            return model
        if self.device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[self.device.index],
                output_device=self.device.index,
            )
        return DistributedDataParallel(model)

    def assert_consistent_state(self, state: Mapping[str, Tensor]) -> str:
        digest = tensor_state_sha256(state)
        digests = self.all_gather_object(digest)
        if len(set(digests)) != 1:
            raise DistributedRunError(
                "model checkpoint state differs across distributed ranks"
            )
        return digest


@contextmanager
def distributed_session(
    runtime: RuntimeConfig,
    *,
    environment: Mapping[str, str] | None = None,
) -> Iterator[DistributedContext]:
    context = DistributedContext.from_environment(runtime, environment=environment)
    context.initialize()
    try:
        yield context
    finally:
        context.close()


class ExactDistributedSampler(Sampler[int]):
    """Rank-strided sampler with exact coverage and no padding or duplication."""

    def __init__(self, dataset: Dataset[Any], *, rank: int, world_size: int) -> None:
        if world_size < 1 or not 0 <= rank < world_size:
            raise DistributedRunError("invalid exact distributed sampler rank/world size")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(cast(Any, self.dataset)), self.world_size))

    def __len__(self) -> int:
        remaining = len(cast(Any, self.dataset)) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size


def training_sampler(
    dataset: Dataset[Any],
    context: DistributedContext,
    *,
    shuffle: bool,
    seed: int,
) -> DistributedSampler[Any] | None:
    if not context.enabled:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )


def prediction_sampler(
    dataset: Dataset[Any], context: DistributedContext
) -> ExactDistributedSampler | None:
    if not context.enabled:
        return None
    return ExactDistributedSampler(dataset, rank=context.rank, world_size=context.world_size)


def set_training_sampler_epoch(loader: Any, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    if not state:
        raise DistributedRunError("model checkpoint state is empty")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not name or not isinstance(tensor, Tensor):
            raise DistributedRunError("model checkpoint state has an invalid entry")
        dense = tensor.detach().cpu().contiguous()
        header = (
            f"{name}\0{dense.dtype}\0{','.join(str(value) for value in dense.shape)}\0"
        ).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        payload = dense.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
