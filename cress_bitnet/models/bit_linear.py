import torch
from torch import Tensor, nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    FairseqDropout,
    LayerNorm)


from numpy.linalg import svd
import math 


class BitLinear(nn.Linear):
    """
    BitLinear is a custom linear layer that performs binarization of weights and quantization of activations
    in a group-wise manner.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default is True.
        num_groups (int, optional): Number of groups to divide the weights and activations into. Default is 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_groups: int = 1,
    ):
        super().__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5
        self.norm = nn.LayerNorm(in_features)
        
        self.g = nn.Parameter(torch.Tensor(in_features))
        self.h = nn.Parameter(torch.Tensor(out_features))    # in_features     out_features
        
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.g, -bound, bound)
        torch.nn.init.uniform_(self.h, -bound, bound)
    def ste(self, x):
        """
        Applies the sign function for binarization and uses Straight-Through Estimator (STE) during backward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Binarized tensor.
        """
        binarized_x = torch.sign(x)
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights_groupwise(self):
        """
        Binarizes the weights of the layer in a group-wise manner using STE.

        Returns:
            Tensor: Binarized weights tensor.
        """
        group_size = self.weight.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]

            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste(weight_group - alpha_g)

        return binarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        """
        Quantizes the activations of the layer in a group-wise manner.

        Args:
            x (Tensor): Input tensor.
            b (int, optional): Number of bits for quantization. Default is 8.

        Returns:
            Tensor: Quantized activations tensor.
        """
        Q_b = 2 ** (b - 1)

        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BitLinear layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Normalize input
        x = self.norm(x)

        # Binarize weights and quantize activations
        

        x = x * self.g
        
        binarized_weights = self.binarize_weights_groupwise()

        # Perform linear transformation
        output = torch.nn.functional.linear(x, binarized_weights, self.bias)

        # Quantize activations
        output = self.quantize_activations_groupwise(output)
        
        output = output * self.h

        # Return output
        return output



def check_string(name, string_list):
    for item in string_list:
        if name in item:
            
            return True
    
    return False
    


def replace_linears_in_hf(
    model,
):
    """
    Replaces all instances of nn.Linear in the given model with BitLinear15b.

    Args:
        model (nn.Module): The model to modify.

    Returns:
        None
    """
    
    check_string_list = ["output_projection", "post_extract_proj"]

    for name, module in model.named_children():
        
        if isinstance(module, nn.Linear) and not check_string(name,check_string_list ):
            # Replace the nn.Linear with BitLinear matching in features and and out_features, and add it to the model

            setattr(
                model,
                name,
                BitLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                ),
            )
        else:
            # Recursively apply to child modules
            replace_linears_in_hf(module)
