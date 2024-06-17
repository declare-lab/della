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

from mergekit.merge_methods.base import MergeMethod
from mergekit.merge_methods.generalized_task_arithmetic import (
    ConsensusMethod,
    GeneralizedTaskArithmeticMerge,
    SparsificationMethod,
)
from mergekit.merge_methods.linear import LinearMerge
from mergekit.merge_methods.passthrough import PassthroughMerge
from mergekit.merge_methods.slerp import SlerpMerge
from mergekit.merge_methods.tokenizer_permute import TokenizerPermutationMerge


def get(method: str) -> MergeMethod:
    if method == "linear":
        return LinearMerge()
    elif method == "slerp":
        return SlerpMerge()
    elif method == "passthrough":
        return PassthroughMerge()
    elif method == "task_arithmetic":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=None,
            sparsification_method=None,
            default_normalize=False,
        )

    elif method == "dare_linear":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=None,
            sparsification_method=SparsificationMethod.rescaled_random,
            default_normalize=False,
        )

    elif method == "dare_ties":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rescaled_random,
            default_normalize=False,
        )
        
    elif method == "ties":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.magnitude,
            default_normalize=True,
        )

    elif method == "della":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rank_magnitude_sampling,
            default_normalize=True,
        )

    elif method == "della_norank":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.magnitude_sampling,
            default_normalize=True,
        )

    elif method == "della_noelect":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rank_magnitude_sampling,
            default_normalize=True,
        )        
    elif method == "della_ada_layer":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rank_magnitude_sampling,
            default_normalize=True,
            lambda_scale = "ada_layer"
        )
        
    elif method == "della_ada_row":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rank_magnitude_sampling,
            default_normalize=True,
            lambda_scale = "ada_row"
        )
        
    elif method == "della_ada_row_inv":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rank_magnitude_sampling,
            default_normalize=True,
            lambda_scale = "ada_row_inv"
        )
        
    elif method == "della_layer_norm":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rank_magnitude_layer_sampling,
            default_normalize=True,
        )

    elif method == "ties_ada_row":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.magnitude,
            default_normalize=True,
            lambda_scale = "ada_row"
        )

    elif method == "ties_ada_row_inv":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.magnitude,
            default_normalize=True,
            lambda_scale = "ada_row_inv"
        )
        
    elif method == "ties_ada_layer":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.magnitude,
            default_normalize=True,
            lambda_scale = "ada_layer"
        )

    elif method == "ties_row":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.magnitude_row,
            default_normalize=True,
        )

    elif method == "dare_della":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rescaled_random,
            default_normalize=True,
        )
    elif method == "dare_della_ada_row":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rescaled_random,
            default_normalize=True,
            lambda_scale = "ada_row"
        )
    elif method == "dare_della_ada_row_inv":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rescaled_random,
            default_normalize=True,
            lambda_scale = "ada_row_inv"
        )
    elif method == "dare_della_ada_layer":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=SparsificationMethod.rescaled_random,
            default_normalize=True,
            lambda_scale = "ada_layer"
        )
    elif method == "nodrop_ada_row":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=None,
            default_normalize=True,
            lambda_scale = "ada_row"
        )
    elif method == "nodrop_ada_row_inv":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=None,
            default_normalize=True,
            lambda_scale = "ada_row_inv"
        )
    elif method == "nodrop_ada_layer":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=ConsensusMethod.sum,
            sparsification_method=None,
            default_normalize=True,
            lambda_scale = "ada_layer"
        )

    elif method == "nodrop_noelect":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=None,
            sparsification_method=None,
            default_normalize=True
        )

    elif method == "dare_noelect":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=None,
            sparsification_method=SparsificationMethod.rescaled_random,
            default_normalize=True,
        )
    elif method == "ties_noelect":
        return GeneralizedTaskArithmeticMerge(
            consensus_method=None,
            sparsification_method=SparsificationMethod.magnitude,
            default_normalize=True,
        )
        
    raise RuntimeError(f"Unimplemented merge method {method}")


__all__ = [
    "MergeMethod",
    "get",
    "LinearMerge",
    "SlerpMerge",
    "PassthroughMerge",
    "GeneralizedTaskArithmeticMerge",
    "TokenizerPermutationMerge",
]
