"""Activation tracker for monitoring layer-wise MLP neuron activations."""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class ActivationTracker:
    """Tracks layer-wise MLP neuron activations for transformer models.

    Monitors the GELU activation output (or fallback c_fc output) across
    all transformer layers, recording per-neuron sequence maximum and
    mean activations without retaining computational graphs.
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize the activation tracker.

        Args:
            model: PyTorch model containing a transformer block structure (model.transformer.h).
        """
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.current: Dict[int, Dict[str, torch.Tensor]] = {}
        self._is_attached: bool = False

    def _make_hook(self, layer_idx: int):
        """Create a forward hook function for a specific layer.

        Args:
            layer_idx: Index of the transformer layer.

        Returns:
            Hook function for PyTorch register_forward_hook.
        """
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            with torch.no_grad():
                tensor = output[0] if isinstance(output, tuple) else output

                # Calculate sequence max and mean activations
                if tensor.dim() == 3:
                    # Shape: (batch, seq_len, features)
                    max_tensor = tensor.max(dim=1).values
                    mean_tensor = tensor.mean(dim=1)
                    if max_tensor.size(0) == 1:
                        max_tensor = max_tensor.squeeze(0)
                        mean_tensor = mean_tensor.squeeze(0)
                elif tensor.dim() == 2:
                    # Shape: (seq_len, features)
                    max_tensor = tensor.max(dim=0).values
                    mean_tensor = tensor.mean(dim=0)
                else:
                    raise ValueError(f"Unexpected activation tensor shape: {tensor.shape}")

                # Store clones detached on CPU to prevent graph retention or memory leaks
                self.current[layer_idx] = {
                    "max": max_tensor.detach().cpu().clone(),
                    "mean": mean_tensor.detach().cpu().clone(),
                }

        return hook

    def attach(self) -> None:
        """Attach forward hooks to all layer MLP activation modules.

        Raises:
            RuntimeError: If attach() is called twice without an intervening detach().
            AttributeError: If model structure does not contain expected transformer blocks.
        """
        if self._is_attached or len(self.handles) > 0:
            raise RuntimeError(
                "ActivationTracker is already attached. Call detach() before calling attach() again."
            )

        blocks = getattr(getattr(self.model, "transformer", None), "h", None)
        if blocks is None:
            raise AttributeError("Model does not possess 'transformer.h' blocks.")

        for i, block in enumerate(blocks):
            mlp = getattr(block, "mlp", None)
            if mlp is None:
                raise AttributeError(f"Layer {i} does not possess an 'mlp' module.")

            act_module = getattr(mlp, "act", None)
            if isinstance(act_module, nn.Module):
                target_module = act_module
            else:
                logger.warning(
                    "Layer %d: block.mlp.act is not an nn.Module; falling back to block.mlp.c_fc "
                    "(activations are pre-GELU).",
                    i,
                )
                c_fc_module = getattr(mlp, "c_fc", None)
                if not isinstance(c_fc_module, nn.Module):
                    raise AttributeError(
                        f"Layer {i}: Neither block.mlp.act nor block.mlp.c_fc is an nn.Module."
                    )
                target_module = c_fc_module

            handle = target_module.register_forward_hook(self._make_hook(i))
            self.handles.append(handle)

        self._is_attached = True

    def detach(self) -> None:
        """Remove all registered hooks and clear stored activation buffers."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.current.clear()
        self._is_attached = False

    def run_and_capture(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        max_length: int = 64,
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """Run a prompt through the model and capture layer-wise activation statistics.

        Args:
            tokenizer: Pretrained tokenizer for encoding the prompt.
            prompt: Text prompt string.
            max_length: Maximum sequence length with truncation.

        Returns:
            Dictionary mapping layer index to {'max': Tensor[3072], 'mean': Tensor[3072]}.

        Raises:
            RuntimeError: If attach() was not called prior to run_and_capture.
        """
        if not self._is_attached or not self.handles:
            raise RuntimeError(
                "ActivationTracker is not attached. Call attach() before running capture."
            )

        self.current.clear()

        device = getattr(self.model, "device", torch.device("cpu"))
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            self.model(**inputs)

        return {layer_idx: dict(stats) for layer_idx, stats in self.current.items()}

    def __enter__(self) -> "ActivationTracker":
        """Enter context manager and attach hooks."""
        self.attach()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and detach hooks."""
        self.detach()
