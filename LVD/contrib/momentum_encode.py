from typing import Iterable
from itertools import zip_longest
import torch


# from stablebaseline3
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/utils.py

@torch.no_grad()
def update_moving_average(ma_model, current_model, beta = 0.01):
    for (n1, current_params), (n2, ma_params) in zip_strict(current_model.state_dict().items(), ma_model.state_dict().items()):
        old_weight, up_weight = ma_params.data, current_params.data

        if "num_batches_tracked" in n1:
            continue

        elif "running_" in n1:
            polyak_update(up_weight, old_weight, 1)

        else:
            polyak_update(up_weight, old_weight, beta)



def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.
    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo


def polyak_update(
    param: torch.Tensor,
    target_param: torch.Tensor,
    tau: float,
) -> None:
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93
    :param params: parameters to use to update the target params
    :param target_params: parameters to update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """

    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        target_param.data.mul_(1 - tau)
        torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
