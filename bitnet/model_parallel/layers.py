import torch
from torch import nn
from torch.nn.parameter import Parameter
from typing import Callable, Optional
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear)


import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from torch.distributed import all_reduce, ReduceOp
from bitnet import BitLinear  # Assuming BitNet provides something like this for quantization

def init_bitnet_weights(tensor, init_method):
    # Initialize using BitNet's recommended method for 1-bit quantized weights
    init_method(tensor)
    return tensor

class BitColumnParallelLinear(nn.Module):
    """Column parallel linear layer with 1-bit quantization using BitNet."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        init_method: callable = init_bitnet_weights,
        stride: int = 1,
        keep_master_weight_for_test: bool = False
    ):
        super(BitColumnParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)

        self.weight = Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        # Use BitNet's quantization initialization method
        self.weight = init_method(self.weight, torch.nn.init.xavier_normal_)

    def forward(self, input_: Tensor) -> Tensor:
        input_parallel = copy_to_model_parallel_region(input_)
        # Apply BitLinear's forward operation, assuming it does something similar
        output_parallel = BitLinear.apply(input_parallel, self.weight, self.bias)  # Pseudocode
        if self.gather_output:
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, gather_output={self.gather_output}'


    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """
    def __init__(
                self, 
                in_features: int, 
                out_features: int, 
                bias: bool = True,
                gather_output: bool = True,
                init_method: Callable[[torch.Tensor], torch.Tensor] = init.xavier_normal_,
                stride: int = 1,
                keep_master_weight_for_test: bool = False,
            ) -> None:
        
        # Keep input parameters
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide_and_check_no_remainder(out_features, world_size)

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        self.weight = Parameter(torch.Tensor(self.output_size_per_partition, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.output_size_per_partition))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

        # Initialize weight.
        self.master_weight = _initialize_affine_weight(
            self.weight,
            self.out_features,
            self.in_features,
            self.output_size_per_partition,
            0,
            init_method,
            stride=stride,
            return_master_weight=keep_master_weight_for_test,
        )

    def get_master_weight(self) -> torch.Tensor:
        return gather_from_model_parallel_region(self.weight.data.transpose(0, 1)).transpose_(0, 1)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output

        


class BitRowParallelLinear(nn.Module):
      """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        in_features: first dimension of matrix A.
        out_features: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
    """
      def __init__():
          pass


