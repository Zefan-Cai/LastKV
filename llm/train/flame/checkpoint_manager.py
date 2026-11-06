import os
import time
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torchtitan.components.checkpoint import CheckpointManager as TorchTitanCheckpointManager
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection


class CheckpointManager(TorchTitanCheckpointManager):
    """Checkpoint manager that avoids unsafe full saves at step 1 and
    tolerates missing keys when resuming from a checkpoint whose model
    architecture has fewer parameters than the current model (e.g. new
    ``q_conv1d`` / ``shortconv`` layers added after the checkpoint was saved).
    """

    def _should_save(self, curr_step: int, force: bool = False) -> bool:
        if not self.enable_checkpoint:
            return False

        if os.environ.get("FLAME_DISABLE_CHECKPOINT_SAVES") == "1":
            return False

        if force:
            return True

        if curr_step % self.interval == 0:
            return True

        return False

    @staticmethod
    def _uses_wrapped_model_namespace(checkpoint_id: str, states: dict[str, Any]) -> bool:
        """Detect DCP checkpoints saved as {"model": model.state_dict()}.

        TorchTitan's model-only initial load expects a flat model state dict, while
        our HF conversion utility historically saved the flat state dict under a
        top-level "model" namespace. Detect that format so pretrained converted
        checkpoints actually hydrate the model instead of being skipped by partial
        load.
        """

        sample_key = next(iter(states), None)
        if sample_key is None:
            return False

        try:
            metadata = dcp.filesystem.FileSystemReader(checkpoint_id).read_metadata()
        except Exception as exc:  # pragma: no cover - best-effort diagnostic path
            logger.warning(
                "Could not read DCP metadata from %s to detect model namespace: %s",
                checkpoint_id,
                exc,
            )
            return False

        checkpoint_keys = metadata.state_dict_metadata.keys()
        return sample_key not in checkpoint_keys and f"model.{sample_key}" in checkpoint_keys

    @staticmethod
    def _checkpoint_state_dict_keys(checkpoint_id: str) -> set[str] | None:
        try:
            metadata = dcp.filesystem.FileSystemReader(checkpoint_id).read_metadata()
        except Exception as exc:  # pragma: no cover - best-effort diagnostic path
            logger.warning(
                "Could not read DCP metadata from %s for post-load hooks: %s",
                checkpoint_id,
                exc,
            )
            return None
        return set(metadata.state_dict_metadata.keys())

    def _run_model_only_post_load_hooks(self, checkpoint_keys: set[str] | None) -> None:
        model_wrapper = self.states.get("model")
        model_parts = getattr(model_wrapper, "model", None)
        if not model_parts:
            return

        hooked = []
        seen = set()
        for model_part in model_parts:
            if not hasattr(model_part, "modules"):
                continue
            for module in model_part.modules():
                hook = getattr(module, "post_initial_model_weight_load", None)
                if not callable(hook) or id(module) in seen:
                    continue
                seen.add(id(module))
                if hook(checkpoint_keys=checkpoint_keys):
                    hooked.append(module.__class__.__name__)

        if hooked:
            logger.info(
                "Ran post-initial model weight load hooks for: %s",
                ", ".join(hooked),
            )

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        """Load checkpoint with ``allow_partial_load=True`` so that new
        parameters absent from the checkpoint are silently skipped instead
        of raising ``RuntimeError: Missing key …``."""

        if self.ft_manager:
            self._ft_load()

        if not self.enable_checkpoint:
            return False

        model_only = False
        if not os.path.exists(self.folder):
            if self.initial_load_path:
                checkpoint_id = self.initial_load_path
                if not os.path.isdir(checkpoint_id):
                    raise ValueError(
                        "initial_load_full_checkpoint is specified but the path is not valid."
                    )
                model_only = self.initial_load_model_weights_only
            else:
                return False
        else:
            if self.initial_load_path:
                logger.info(
                    "`initial_load_path` is provided but the checkpoint folder exists. "
                    "Checkpointer will use the checkpoints from the checkpoint folder."
                )
            step = self._find_load_step() if step == -1 else step
            if step == -1:
                return False
            model_only = step == 0
            checkpoint_id = self._create_checkpoint_id(step)

            if not os.path.isdir(checkpoint_id):
                return False

        logger.info(f"Loading the checkpoint from {checkpoint_id}.")
        begin = time.monotonic()
        states = self._states_to_load(model_only)
        if model_only and self._uses_wrapped_model_namespace(checkpoint_id, states):
            logger.info(
                "Detected model-prefixed model-only checkpoint; loading initial "
                "weights through top-level 'model' namespace."
            )
            states = {"model": states}
        dcp.load(
            states,
            checkpoint_id=checkpoint_id,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )
        if model_only:
            self._run_model_only_post_load_hooks(
                self._checkpoint_state_dict_keys(checkpoint_id),
            )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        return True
