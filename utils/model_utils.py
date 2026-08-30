"""Base model utilities with frozen parameter support."""

import paddle
import paddle.nn as nn
from typing import List, Optional, Union


def freeze_params(model: nn.Layer, param_names: Optional[Union[str, List[str]]] = None) -> int:
    """Freeze parameters by name or pattern.

    Args:
        model: Paddle model.
        param_names: Parameter name(s) or pattern(s) to freeze.
                     If None, freeze all parameters.

    Returns:
        Number of parameter tensors frozen.

    Examples:
        freeze_params(model, 'encoder')  # Freeze all params with 'encoder' in name
        freeze_params(model, ['encoder', 'timbre'])  # Freeze multiple patterns
        freeze_params(model)  # Freeze all
    """
    if param_names is None:
        # Freeze all
        frozen_count = 0
        for param in model.parameters():
            param.stop_gradient = True
            frozen_count += 1
        return frozen_count

    # Convert to list
    if isinstance(param_names, str):
        param_names = [param_names]

    # Freeze matching parameters
    frozen_count = 0
    for name, param in model.named_parameters():
        for pattern in param_names:
            if pattern in name:
                param.stop_gradient = True
                frozen_count += 1
                break

    return frozen_count


def unfreeze_params(model: nn.Layer, param_names: Optional[Union[str, List[str]]] = None):
    """Unfreeze parameters by name or pattern.

    Args:
        model: Paddle model.
        param_names: Parameter name(s) or pattern(s) to unfreeze.
                     If None, unfreeze all parameters.
    """
    if param_names is None:
        # Unfreeze all
        for param in model.parameters():
            param.stop_gradient = False
        return

    # Convert to list
    if isinstance(param_names, str):
        param_names = [param_names]

    # Unfreeze matching parameters
    unfrozen_count = 0
    for name, param in model.named_parameters():
        for pattern in param_names:
            if pattern in name:
                param.stop_gradient = False
                unfrozen_count += 1
                break

    print(f"Unfrozen {unfrozen_count} parameters matching patterns: {param_names}")


def get_trainable_params(model: nn.Layer) -> List[paddle.Tensor]:
    """Get list of trainable (non-frozen) parameters.

    Args:
        model: Paddle model.

    Returns:
        List of trainable parameters.
    """
    return [p for p in model.parameters() if not p.stop_gradient]


def count_parameters(model: nn.Layer, trainable_only: bool = False) -> int:
    """Count model parameters.

    Args:
        model: Paddle model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        params = get_trainable_params(model)
    else:
        params = model.parameters()

    return sum(p.numel() for p in params)


def print_model_summary(model: nn.Layer):
    """Print model parameter summary.

    Args:
        model: Paddle model.
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    frozen_params = total_params - trainable_params

    print(f"\n{'='*60}")
    print(f"Model Parameters Summary")
    print(f"{'='*60}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {frozen_params:,}")
    print(f"{'='*60}\n")


def load_pretrained_with_frozen(
    model: nn.Layer,
    pretrained_path: str,
    frozen_params: Optional[Union[str, List[str]]] = None,
    strict: bool = True,
):
    """Load pretrained checkpoint and optionally freeze parameters.

    Useful for fine-tuning scenarios where you want to load a pretrained
    model and freeze certain components (e.g., content encoder).

    Args:
        model: Paddle model to load into.
        pretrained_path: Path to checkpoint (.pdparams).
        frozen_params: Parameter patterns to freeze after loading.
        strict: If True, require exact parameter match.

    Example:
        # Load pretrained content extractor and freeze it
        load_pretrained_with_frozen(
            model.content_extractor,
            'ckpts/content_extractor/best.pdparams',
            frozen_params=['emformer', 'mel_proj']
        )
    """
    # Load checkpoint
    state_dict = paddle.load(pretrained_path)
    model.set_state_dict(state_dict)
    print(f"Loaded pretrained weights from: {pretrained_path}")

    # Freeze specified parameters
    if frozen_params is not None:
        freeze_params(model, frozen_params)

    # Print summary
    print_model_summary(model)
