import torch 
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from util.misc import inverse_sigmoid
from modules.transformer import PreProccessor, TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder 
import copy
from typing import Optional, List



class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.state_classifier = None
        self.obj_classifier = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                reference_points: Optional[Tensor] = None):
        output = tgt

        intermediate = []
        intermediate_attn = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            output, attn_maps = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.state_classifier is not None:
                tmp = self.state_classifier[lid](output)
                new_reference_point = tmp + inverse_sigmoid(reference_points)
                new_reference_point = new_reference_point.sigmoid()
                reference_points = new_reference_point.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                intermediate_reference_points.append(reference_points)

            intermediate_attn.append(attn_maps)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_attn), torch.stack(intermediate_reference_points)

        return output.unsqueeze(0), torch.stack(intermediate_attn), reference_points.unsqueeze(0)
# --------------------------------------------- #


# -------------- HELPER FUNCTIONS ------------- #
def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
# --------------------------------------------- #