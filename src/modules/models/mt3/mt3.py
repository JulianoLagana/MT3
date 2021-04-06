import torch
from torch import nn
from modules.position_encoder import LearnedPositionEncoder
from modules.mlp import MLP
from modules.models.mt3.transformer import TransformerEncoder, TransformerDecoder, PreProccessor, TransformerEncoderLayer, TransformerDecoderLayer
from modules.contrastive_classifier import ContrastiveClassifier
from util.misc import NestedTensor, inverse_sigmoid
import copy
import math


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MOTT(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.d_detections = params.arch.d_detections
        self.temporal_encoder = LearnedPositionEncoder(params.data_generation.n_timesteps, params.arch.d_model)
        self.measurement_normalization_factor = params.data_generation.field_of_view_ub - params.data_generation.field_of_view_lb
        self.preprocessor = PreProccessor(params.arch.d_model,
                                          params.arch.d_detections,
                                          normalization_constant=self.measurement_normalization_factor)
        encoder_layer = TransformerEncoderLayer(params.arch.d_model,
                                                nhead=params.arch.encoder.n_heads,
                                                dim_feedforward=params.arch.encoder.dim_feedforward,
                                                dropout=params.arch.encoder.dropout,
                                                activation="relu",
                                                normalize_before=False)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=params.arch.encoder.n_layers, norm=None)
        decoder_layer = TransformerDecoderLayer(params.arch.d_model,
                                                nhead=params.arch.decoder.n_heads,
                                                dim_feedforward=params.arch.decoder.dim_feedforward,
                                                dropout=params.arch.decoder.dropout,
                                                activation="relu",
                                                normalize_before=False)
        decoder_norm = nn.LayerNorm(params.arch.d_model) if params.loss.return_intermediate else None
        self.decoder = TransformerDecoder(decoder_layer,
                                          num_layers=params.arch.decoder.n_layers,
                                          norm=decoder_norm,
                                          return_intermediate=params.loss.return_intermediate)
        
        self.query_embed = nn.Embedding(params.arch.num_queries, params.arch.d_model)

        self.state_classifier = MLP(params.arch.d_model,
                                    hidden_dim=params.arch.d_prediction_hidden,
                                    output_dim=params.arch.d_detections,
                                    num_layers=params.arch.n_prediction_layers)
        self.obj_classifier = nn.Linear(params.arch.d_model, 1)
        self.aux_loss = params.loss.return_intermediate

        if self.params.loss.contrastive_classifier:
            self.contrastive_classifier = ContrastiveClassifier(params.arch.d_model)

        if self.params.loss.false_classifier:
            self.false_classifier = MLP(params.arch.d_model,
                                        hidden_dim=params.arch.d_prediction_hidden,
                                        output_dim=1,
                                        num_layers=1)

        self.with_state_refine = params.arch.with_state_refine
        self.two_stage = params.arch.two_stage
        self.d_model = params.arch.d_model

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (self.decoder.num_layers + 1) if self.two_stage else self.decoder.num_layers
        if self.with_state_refine:
            # Predict zero delta state
            nn.init.constant_(self.state_classifier.layers[-1].weight.data, 0)
            nn.init.constant_(self.state_classifier.layers[-1].bias.data, 0)
            self.obj_classifier = _get_clones(self.obj_classifier, num_pred)
            self.state_classifier = _get_clones(self.state_classifier, num_pred)
            self.decoder.state_classifier = self.state_classifier
        else:
            self.obj_classifier = nn.ModuleList([self.obj_classifier for _ in range(num_pred)])
            self.state_classifier = nn.ModuleList([self.state_classifier for _ in range(num_pred)])
            self.decoder.state_classifier = None

        if self.two_stage:
            # hack implementation for two-stage
            self.decoder.obj_classifier = self.obj_classifier

            self.enc_output = nn.Linear(params.arch.d_model, params.arch.d_model)
            self.enc_output_norm = nn.LayerNorm(params.arch.d_model)

            self.pos_trans = nn.Linear(self.d_model * 2, self.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(self.d_model * 2)

            self.num_queries = params.arch.num_queries
        else:
            self.reference_points_linear = nn.Linear(params.arch.d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if not self.two_stage:
            nn.init.xavier_uniform_(self.reference_points_linear.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points_linear.bias.data, 0.)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, measurements):
        # Set proposed state as measurement location
        output_proposals_valid = ((measurements > 0.01) & (measurements < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(measurements/(1-measurements))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        # Init out memory
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        # Project encodings
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals
    
    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = self.d_model*2/self.d_detections
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, ndim
        proposals = proposals.sigmoid() * scale
        # N, L, dim, num_pos_feats
        pos = proposals[:, :, :, None] / dim_t
        # N, L, dim, num_pos_feats/ndim, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        # N, L, num_pos_feats*dim
        return pos

    def forward(self, samples: NestedTensor, unique_ids):
        mapped_time_idx = torch.round(samples.tensors[:,:,-1] / self.params.data_generation.dt)
        time_encoding = self.temporal_encoder(mapped_time_idx.long())
        # bs, n_timesteps, d_detections + 1
        src = self.preprocessor(samples.tensors[:,:,:self.d_detections])
        
        mask = samples.mask

        bs, num_batch_max_meas, d_detections = src.shape
        src = src.permute(1, 0, 2)
        time_encoding = time_encoding.permute(1, 0, 2)

        query_embed = self.query_embed.weight
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        
        # Encoder
        memory = self.encoder(src, src_key_padding_mask=mask, pos=time_encoding)


        aux_classifications = {}
        if self.params.loss.contrastive_classifier:
            contrastive_classifications = self.contrastive_classifier(memory.permute(1, 0, 2), padding_mask=mask)
            aux_classifications['contrastive_classifications'] = contrastive_classifications

        if self.params.loss.false_classifier:
            false_classifications = self.false_classifier(memory)
            aux_classifications['false_classifications'] = false_classifications

        # prepare memory for decoder
        _, _, c = memory.shape
        if self.two_stage:
            normalized_meas = samples.tensors[:,:,:self.d_detections]/self.measurement_normalization_factor+0.5
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory.permute(1, 0, 2), mask, measurements=normalized_meas)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.obj_classifier[self.decoder.num_layers](output_memory)
            # Set masked predictions to "0" probability
            enc_outputs_class = enc_outputs_class.masked_fill(mask.unsqueeze(-1), -100_000_000)
            enc_outputs_coord_unact = self.state_classifier[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.num_queries

            topk_proposals_indices = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals_indices.unsqueeze(-1).repeat(1, 1, self.d_detections))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            reference_points = reference_points.permute(1, 0, 2)

            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)

            query_embed = query_embed.permute(1, 0, 2)
            tgt = tgt.permute(1, 0, 2)
        else:
            tgt = torch.zeros_like(query_embed)
            reference_points = self.reference_points_linear(query_embed).sigmoid()

        init_reference = reference_points.permute(1, 0, 2)

        hs, attn_maps, inter_references = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=time_encoding, query_pos=query_embed, reference_points=reference_points)
        hs = hs.transpose(1,2)
        pred_obj = []
        pred_state = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1].permute(1, 0, 2)
            
            reference = inverse_sigmoid(reference)
            predicted_obj_prob = self.obj_classifier[lvl](hs[lvl])
            tmp = self.state_classifier[lvl](hs[lvl])
            tmp = tmp + reference
            predicted_state = (tmp.sigmoid() - 0.5)*self.measurement_normalization_factor
            pred_obj.append(predicted_obj_prob)
            pred_state.append(predicted_state)

        pred_obj = torch.stack(pred_obj)
        pred_state = torch.stack(pred_state)

        out = {'state': pred_state[-1], 'logits': pred_obj[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(pred_obj, pred_state)

        if self.two_stage:
            enc_outputs_coord = (enc_outputs_coord_unact.sigmoid() - 0.5)*self.measurement_normalization_factor
            # Make sure padded measurements cannot be matched/are far away
            enc_outputs_coord = enc_outputs_coord.masked_fill(mask.unsqueeze(-1).repeat(1,1,enc_outputs_coord.shape[-1]), self.measurement_normalization_factor*5)
            out['enc_outputs'] = {'logits': enc_outputs_class, 'state': enc_outputs_coord}

        memory = memory.permute(1, 0, 2)
        return out, memory, aux_classifications, hs, attn_maps.permute(1,0,2,3)

    @torch.jit.unused
    def _set_aux_loss(self, pred_obj, pred_state):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'logits': a, 'state': b}
                for a, b in zip(pred_obj[:-1], pred_state[:-1])]

    def to(self, device):
        super().to(device)
        if self.params.loss.contrastive_classifier:
            self.contrastive_classifier.to(device)
        
