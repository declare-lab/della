# Copyright (C) 2024 Charles O. Goddard
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from enum import Enum

import torch


class SparsificationMethod(str, Enum):
    magnitude = "magnitude"

    magnitude_row = "magnitude_row"

    rank_magnitude_sampling = "rank_magnitude_sampling"

    rank_magnitude_layer_sampling = "rank_magnitude_layer_sampling"

    magnitude_sampling = "magnitude_sampling"

    rescaled_random = "rescaled_random"


def magnitude(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor

    k = int(density * tensor.view(-1).shape[0])

    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.topk(w, k=k, largest=True)
    mask.view(-1)[topk.indices] = 1

    return tensor * mask

def magnitude_row_wise(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Masks out the smallest values row-wise, retaining a proportion of `density`."""
    if density >= 1:
        return tensor

    assert 0 < density <= 1, "Density must be between 0 and 1"

    if len(tensor.shape)<2:
        tensor = tensor.unsqueeze(0)

    rows, cols = tensor.shape
    k = int(density * cols)
    assert k > 0, "Not gonna zero out the whole row buddy"

    # Compute the indices of the top k elements in each row
    w = tensor.abs()
    topk = torch.topk(w, k=k, dim=1, largest=True)

    # Create a mask using the topk indices
    mask = torch.zeros_like(tensor).scatter(1, topk.indices, 1)

    return (tensor * mask).squeeze(0)

def bernoulli(
    tensor: torch.Tensor, density: float, rescale: bool = True
) -> torch.Tensor:
    if density >= 1:
        return tensor

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        # torch.bernoulli not implemented for float16 on CPU, upcast to float32
        work_dtype = torch.float32

    mask = torch.bernoulli(
        torch.full_like(input=tensor, fill_value=density, dtype=work_dtype)
    )
    res = tensor.to(work_dtype) * mask
    if rescale:
        res /= density
    return res.to(tensor.dtype)

def rank_magnitude(
    tensor: torch.Tensor, density: float, rescale: bool = True, epsilon: float = 0.05
) -> torch.Tensor:
    if density >= 1:
        return tensor

    if density <= epsilon or density>=(1-epsilon):
        print("Density out of Bounds")

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        # torch.bernoulli not implemented for float16 on CPU, upcast to float32
        work_dtype = torch.float32

    if len(tensor.shape)<2:
        tensor = tensor.unsqueeze(0)
    
    # Get Rank matrix
    tensor_abs = torch.abs(tensor)

    sorted_indices = torch.argsort(tensor_abs, dim=1, descending=False)
    
    ranking_tensor = torch.zeros_like(tensor_abs, dtype=work_dtype)
    for i in range(tensor_abs.size(0)):
        ranking_tensor[i][sorted_indices[i]] = torch.arange(1, tensor.size(1) + 1, dtype= work_dtype).to(tensor.device)
    
    range_vals = ranking_tensor.max(dim = 1, keepdim=True).values - ranking_tensor.min(dim = 1, keepdim=True).values 
    norm_metrics = (ranking_tensor - ranking_tensor.min(dim = 1, keepdim=True).values)/(range_vals)
    final_probabilities = (density-epsilon) + norm_metrics * (2*epsilon)
    mask = torch.bernoulli(final_probabilities).to(work_dtype)

    res = tensor.to(work_dtype) * mask

    if rescale:
        res = res / (final_probabilities.to(work_dtype))
        assert not torch.isnan(res).any()
        assert not torch.isinf(res).any()

    return res.squeeze(0)

def layer_rank_matrix(matrix):
    flattened_matrix = matrix.flatten()

    sorted_indices = torch.argsort(flattened_matrix, descending=False)

    ranking_tensor = torch.zeros_like(flattened_matrix, dtype=torch.long)
    ranking_tensor[sorted_indices] = torch.arange(1, flattened_matrix.size(0) + 1).to(matrix.device)

    return ranking_tensor.reshape(matrix.shape)

def rank_magnitude_layer(
    tensor: torch.Tensor, density: float, rescale: bool = True, epsilon: float = 0.05
) -> torch.Tensor:
    if density >= 1:
        return tensor

    if density <= epsilon or density>=(1-epsilon):
        print("Density out of Bounds")

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        # torch.bernoulli not implemented for float16 on CPU, upcast to float32
        work_dtype = torch.float32

    if len(tensor.shape)<2:
        tensor = tensor.unsqueeze(0)
    
    # Get Rank matrix
    tensor_abs = torch.abs(tensor)
    ranking_tensor = layer_rank_matrix(tensor_abs)
    
    range_vals = ranking_tensor.max() - ranking_tensor.min() 
    norm_metrics = (ranking_tensor - ranking_tensor.min())/(range_vals)
    final_probabilities = (density-epsilon) + norm_metrics * (2*epsilon)

    mask = torch.bernoulli(final_probabilities).to(work_dtype)
    res = tensor.to(work_dtype) * mask

    if rescale:
        res = res / (final_probabilities.to(work_dtype))
        assert not torch.isnan(res).any()
        assert not torch.isinf(res).any()

    return res.squeeze(0)


def magnitude_sample(
    tensor: torch.Tensor, density: float, rescale: bool = True, epsilon=0.05
) -> torch.Tensor:
    if density >= 1:
        return tensor

    if density <= epsilon or density>=(1-epsilon):
        print("Density out of Bounds")

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        # torch.bernoulli not implemented for float16 on CPU, upcast to float32
        work_dtype = torch.float32
    if len(tensor.shape)<2:
        tensor = tensor.unsqueeze(0)
    
    # Get Rank matrix
    tensor_abs = torch.abs(tensor)

    range_vals = tensor_abs.max(dim = 1, keepdim=True).values - tensor_abs.min(dim = 1, keepdim=True).values +1e-7
    norm_metrics = (tensor_abs - tensor_abs.min(dim = 1, keepdim=True).values)/(range_vals)
    final_probabilities = (density-epsilon) + norm_metrics * (2*epsilon)

    mask = torch.bernoulli(final_probabilities).to(work_dtype)
    res = tensor.to(work_dtype) * mask

    if rescale:
        res = res / (final_probabilities.to(work_dtype))
        assert not torch.isnan(res).any()
        assert not torch.isinf(res).any()

    return res.squeeze(0)


def sparsify(
    tensor: torch.Tensor, density: float, method: SparsificationMethod, epsilon:float, rescale: bool = True
) -> torch.Tensor:
    if method == SparsificationMethod.magnitude:
        return magnitude(tensor, density=density)
        
    if method == SparsificationMethod.magnitude_row:
        return magnitude_row_wise(tensor, density=density)
        
    elif method == SparsificationMethod.rescaled_random:
        return bernoulli(tensor, density=density, rescale=rescale)

    elif method == SparsificationMethod.rank_magnitude_sampling:
        return rank_magnitude(tensor, density=density, rescale=rescale, epsilon=epsilon)

    elif method == SparsificationMethod.rank_magnitude_layer_sampling:
        return rank_magnitude_layer(tensor, density=density, rescale=rescale, epsilon=epsilon)

    elif method == SparsificationMethod.magnitude_sampling:
        return magnitude_sample(tensor, density=density, rescale=rescale, epsilon=epsilon)

    else:
        raise NotImplementedError(method)
