import os
import contextlib
import logging
from typing import Any, Dict, Iterator, List, Optional, Set, Type

import torch
import torch.nn as nn

import determined as det
from determined import profiler, pytorch
from determined.common import check
from determined.tensorboard import get_base_path
from deepspeed import PipelineEngine

# DeepSpeed supports mixed precision through NVidia Apex AMP.  ZeRO optimizer requires Apex AMP
# and cannot be used with more complex AMP modes.
try:
    import apex
except ImportError:
    if torch.cuda.is_available():
        logging.warning("Failed to import apex.")
    pass

class DeepSpeedTrialContext(det.TrialContext, pytorch._PyTorchReducerContext):
    """Contains runtime information for any Determined workflow that uses the ``DeepSpeed`` API.

    With this class, users can do the following things:

    1. Wrap DeepSpeed model engines, optimizers, and LR schedulers with their Determined-compatible
       counterparts using :meth:`wrap_model`, :meth:`wrap_optimizer`, :meth:`wrap_lr_scheduler`,
       respectively. The Determined-compatible objects are capable of transparent
       distributed training, checkpointing and exporting, mixed-precision training,
       and gradient aggregation.
    2. Configure apex amp by calling :meth:`configure_apex_amp` (optional).
    3. Calculate the gradients with :meth:`backward` on a specified loss.
    4. Run an optimization step with :meth:`step_optimizer`.
    5. Functionalities inherited from :class:`determined.TrialContext`, including getting
       the runtime information and properly handling training data in distributed training.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        det.TrialContext.__init__(self, *args, **kwargs)
        pytorch._PyTorchReducerContext.__init__(self, self.distributed._zmq_allgather)

        self._init_device()

        # Track which types we have issued warnings for in to_device().
        self._to_device_warned_types = set()  # type: Set[Type]

        # The following attributes are initialized during the lifetime of
        # a PyTorchTrialContext.
        self.models = []  # type: List[nn.Module]
        self.profiler = None  # type: Any
        self._epoch_len = None  # type: Optional[int]

        self._scaler = None
        self._use_apex = False
        self._loss_ids = {}  # type: Dict[torch.Tensor, int]
        self._last_backward_batch_idx = None  # type: Optional[int]
        self._current_batch_idx = None  # type: Optional[int]

        self.experimental = pytorch.PyTorchExperimentalContext(self)
        self._reducers = pytorch._PyTorchReducerContext()
        self._determined_profiler = None  # type: Optional[profiler.ProfilerAgent]
        self.mpu = pytorch.deepspeed.ModelParallelUnit(self.distributed)
        self.train_micro_batch_size_per_gpu = None
        self.uses_pipeline_parallelism = False

    def wrap_model_engine(self, model: torch.nn.Module) -> torch.nn.Module:
        """Returns a wrapped model."""

        if self.env.managed_training:
            model = model.to(self.device)
        if self.train_micro_batch_size_per_gpu is None:
            self.train_micro_batch_size_per_gpu = model.train_micro_batch_size_per_gpu()
        else:
            assert self.train_micro_batch_size_per_gpu==model.train_micro_batch_size_per_gpu(), "micro batch size do not match across DeepSpeed model engines."
        self.models.append(model)
        if isinstance(model, PipelineEngine):
            self.uses_pipeline_parallelism = True
        return model

    def get_num_micro_batches_per_batch(self) -> int:
        return self.get_global_batch_size() // self.mpu.get_data_parallel_world_size() // self.train_micro_batch_size_per_gpu

    def wrap_mpu(self, mpu: pytorch.deepspeed.ModelParallelUnit):
        self.mpu = mpu

    def set_profiler(self, *args: List[str], **kwargs: Any) -> None:
        """
        Sets a torch profiler instance on the trial context to be called in _pytorch_trial
        when training.
        """
        self.profiler = torch.profiler.profile(
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(get_base_path({}))),
            *args,
            **kwargs,
        )

    def _set_determined_profiler(self, prof: profiler.ProfilerAgent) -> None:
        self._determined_profiler = prof

    @contextlib.contextmanager
    def _record_timing(self, metric_name: str, accumulate: bool = False) -> Iterator[None]:
        if not self._determined_profiler:
            yield
            return
        with self._determined_profiler.record_timing(metric_name, accumulate):
            yield

    def _init_device(self) -> None:
        self.n_gpus = len(self.env.container_gpus)
        if self.distributed.size > 1:
            if self.n_gpus > 0:
                # We launch a horovod process per GPU. Each process
                # needs to bind to a unique GPU.
                self.device = torch.device("cuda", os.environ.get("LOCAL_RANK"))
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")
        elif self.n_gpus > 0:
            self.device = torch.device("cuda", 0)
        else:
            self.device = torch.device("cpu")
        check.is_not_none(self.device)

    def to_device(self, data: pytorch._Data) -> pytorch.TorchData:
        """Map generated data to the device allocated by the Determined cluster.

        All the data in the data loader and the models are automatically moved to the
        allocated device. This method aims at providing a function for the data generated
        on the fly.
        """
        with self._record_timing("to_device", accumulate=True):
            return pytorch.to_device(data, self.device, self._to_device_warned_types)

    def is_epoch_start(self) -> bool:
        """
        Returns true if the current batch is the first batch of the epoch.

        .. warning::
            Not accurate for variable size epochs.
        """
        if self._current_batch_idx is None:
            raise det.errors.InternalException("Training hasn't started.")
        if self._epoch_len is None:
            raise det.errors.InternalException("Training DataLoader uninitialized.")
        return self._current_batch_idx % self._epoch_len == 0

    def is_epoch_end(self) -> bool:
        """
        Returns true if the current batch is the last batch of the epoch.

        .. warning::
            Not accurate for variable size epochs.
        """
        if self._current_batch_idx is None:
            raise det.errors.InternalException("Training hasn't started.")
        if self._epoch_len is None:
            raise det.errors.InternalException("Training DataLoader uninitialized.")
        return self._current_batch_idx % self._epoch_len == self._epoch_len - 1

    def current_train_epoch(self) -> int:
        if self._current_batch_idx is None:
            raise det.errors.InternalException("Training hasn't started.")
        if self._epoch_len is None:
            raise det.errors.InternalException("Training DataLoader uninitialized.")
        return self._current_batch_idx // self._epoch_len

    def current_train_batch(self) -> int:
        """
        Current global batch index
        """
        if self._current_batch_idx is None:
            raise det.errors.InternalException("Training hasn't started.")
        return self._current_batch_idx
