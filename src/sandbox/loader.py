"""Restricted offline model loader for NeuroFence."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def compute_model_hash(model_dir: Union[str, Path]) -> str:
    """Calculate a single SHA-256 digest over all .safetensors weight files.

    Note:
        This fingerprints the model's weight files (.safetensors) only. It does
        not include configuration files, tokenizers, generation configs, or other
        auxiliary metadata.

    Args:
        model_dir: Path to the model directory.

    Returns:
        Hex-encoded SHA-256 digest string.

    Raises:
        FileNotFoundError: If the directory does not exist or no .safetensors files are found.
    """
    path = Path(model_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {path}")

    safetensors_files = sorted(path.glob("*.safetensors"), key=lambda p: p.name)
    if not safetensors_files:
        raise FileNotFoundError(f"No .safetensors weight files found in '{path}'.")

    hasher = hashlib.sha256()
    chunk_size = 65536  # 64 KB chunks
    for file_path in safetensors_files:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                hasher.update(chunk)

    return hasher.hexdigest()


def load_model_sandboxed(
    model_dir: Union[str, Path],
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, Dict[str, Any]]:
    """Load a pretrained model and tokenizer offline with restricted local parameters.

    IMPORTANT SECURITY NOTE:
        This is a restricted local loader designed for offline execution without
        remote code execution or arbitrary pickle loading; it is NOT an operating-system
        security sandbox. Operating system-level isolation (such as containers,
        virtual machines, or seccomp/AppArmor policies) must be provided externally
        if untrusted execution containment is required.

    Args:
        model_dir: Path to the model directory.

    Returns:
        Tuple of (model, tokenizer, metadata):
            - model: Hugging Face PreTrainedModel on CPU in eval mode.
            - tokenizer: PreTrainedTokenizerBase tokenizer instance.
            - metadata: Dictionary containing model details and weight digest.

    Raises:
        FileNotFoundError: If model_dir or config.json does not exist.
        ValueError: If the model architecture is not supported (only 'gpt2' is supported).
    """
    path = Path(model_dir).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {path}")

    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in model directory: {path}")

    # Load configuration strictly offline without remote code execution
    config = AutoConfig.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
    )

    # For this first version, support only gpt2 architecture (including distilgpt2)
    if getattr(config, "model_type", None) != "gpt2":
        raise ValueError(
            f"Unsupported model architecture '{getattr(config, 'model_type', None)}'. "
            "NeuroFence currently supports only 'gpt2' models (including distilgpt2)."
        )

    # Compute fingerprint over .safetensors weights
    model_hash = compute_model_hash(path)

    # Load tokenizer strictly offline without remote code execution
    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
    )

    # Load model strictly offline using safetensors without pickle fallback
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    )

    # Enforce CPU execution and set evaluation mode
    model.to("cpu")
    model.eval()

    metadata: Dict[str, Any] = {
        "path": str(path),
        "hash_sha256": model_hash,
        "model_type": config.model_type,
        "num_layers": getattr(config, "n_layer", getattr(config, "num_hidden_layers", None)),
        "hidden_size": getattr(config, "n_embd", getattr(config, "hidden_size", None)),
        "vocab_size": getattr(config, "vocab_size", len(tokenizer)),
    }

    return model, tokenizer, metadata


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    clean_model_dir = project_root / "models" / "clean_model"

    _, _, metadata = load_model_sandboxed(clean_model_dir)
    print(json.dumps(metadata, indent=2))
    print("Local model loaded successfully.")