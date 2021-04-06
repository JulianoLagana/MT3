import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def check_gospa_parameters(c, p, alpha):
    """ Check parameter bounds.

    If the parameter values are outside the allowable range specified in the
    definition of GOSPA, a ValueError is raised.
    """
    if alpha <= 0 or alpha > 2:
        raise ValueError("The value of alpha is outside the range (0, 2]")
    if c <= 0:
        raise ValueError("The cutoff distance c is outside the range (0, inf)")
    if p < 1:
        raise ValueError("The order p is outside the range [1, inf)")

#order, cutoff_distance, alpha=2, num_classes=1, device='cpu'
class MotLoss(nn.Module):
    def __init__(self, params):
        check_gospa_parameters(params.loss.cutoff_distance, params.loss.order, params.loss.alpha)
        super().__init__()
        self.params = params
        self.order = params.loss.order
        self.cutoff_distance = params.loss.cutoff_distance
        self.alpha = params.loss.alpha
        self.miss_cost = self.cutoff_distance ** self.order
        self.device = torch.device(params.training.device)
        if params.arch.num_queries == params.data_generation.avg_gt_objects:
            weight = 1  # deactivate weighing in cases where it would break training
        else:
            weight = (params.arch.num_queries-params.data_generation.avg_gt_objects)/params.data_generation.avg_gt_objects
        pos_weight = torch.ones([1])*weight # binary classification
        self.detr_logits_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.to(self.device)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                              for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def compute_hungarian_matching(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        output_state = outputs['state']
        output_logits = outputs['logits'].sigmoid().flatten(0,1)

        bs, num_queries = output_state.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, d_label]
        out = output_state.flatten(0, 1)

        # Also concat the target labels
        # [sum(num_objects), d_labels]
        tgt = torch.cat(targets)

        # Compute the L2 cost
        # [batch_size * num_queries, sum(num_objects)]
        cost = torch.pow(input=torch.cdist(out, tgt, p=2), exponent=self.order)
        cost -= output_logits

        # Reshape
        # [batch_size, num_queries, sum(num_objects)]
        #cost = cost.view(bs, num_queries, -1)
        cost = cost.view(bs, num_queries, -1).cpu()

        # List with num_objects for each training-example
        sizes = [len(v) for v in targets]

        # Perform hungarian matching using scipy linear_sum_assignment
        with torch.no_grad():
            indices = [linear_sum_assignment(
                c[i]) for i, c in enumerate(cost.split(sizes, -1))]
            permutation_idx = [(torch.as_tensor(i, dtype=torch.int64).to(torch.device(self.device)), torch.as_tensor(
                j, dtype=torch.int64).to(self.device)) for i, j in indices]

        return permutation_idx, cost.to(self.device)
    
    def compute_orig_gospa_matching(self, outputs, targets, existance_threshold):
        """ Performs the matching. Note that this can NOT be used as a loss function

        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

            existance_threshold: Float in range (0,1) that decides which object are considered alive and which are not. 

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        assert 'state' in outputs, "'state' should be in dict"
        assert 'logits' in outputs, "'logits' should be in dict"
        assert self.order == 1, 'This code does not work for loss.order != 1'
        assert self.alpha == 2, 'The permutation -> assignment relation used to decompose GOSPA might require that loss.alpha == 2'

        output_state = outputs['state'].detach()
        output_logits = outputs['logits'].sigmoid().detach()

        bs, num_queries = output_state.shape[:2]
        loss = torch.zeros(size=(1,))
        localization_cost = 0
        missed_target_cost = 0
        false_target_cost = 0
        indices = []

        for i in range(bs):
            alive_idx = output_logits[i, :].squeeze(-1) > existance_threshold
            alive_output = output_state[i, alive_idx, :]
            current_targets = targets[i]
            permutation_length = 0

            if len(current_targets) == 0:
                indices.append(([], []))
                loss += torch.Tensor([self.miss_cost/self.alpha * len(alive_output)])
                false_target_cost = self.miss_cost/self.alpha * len(alive_output)
            elif len(alive_output) == 0:
                indices.append(([], []))
                loss += torch.Tensor([self.miss_cost/self.alpha * len(current_targets)])
                missed_target_cost = self.miss_cost / self.alpha * len(current_targets)
            else:
                dist = torch.cdist(alive_output, current_targets, p=2)
                dist = dist.clamp_max(self.cutoff_distance)
                c = torch.pow(input=dist, exponent=self.order)
                c = c.cpu()
                target_idx, output_idx = linear_sum_assignment(c)
                indices.append((target_idx, output_idx))

                for t, o in zip(target_idx, output_idx):
                    loss += c[t,o]
                    if c[t, o] < self.cutoff_distance:
                        localization_cost += c[t, o].item()
                        permutation_length += 1
                
                cardinality_error = abs(len(alive_output) - len(current_targets))
                loss += self.miss_cost/self.alpha * cardinality_error

                missed_target_cost += (len(targets[0]) - permutation_length) * (self.miss_cost/self.alpha)
                false_target_cost += (len(alive_output) - permutation_length) * (self.miss_cost/self.alpha)

        decomposition = {'localization': localization_cost, 'missed': missed_target_cost, 'false': false_target_cost,
                         'n_matched_objs': permutation_length}
        return loss, indices, decomposition

    def compute_prob_gospa_matching(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        output_state = outputs['state']
        output_logits = outputs['logits'].sigmoid()

        bs, num_queries = output_state.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_queries, d_label]
        out = output_state.flatten(0, 1)
        probs = output_logits.flatten(0, 1)
        # Also concat the target labels
        # [sum(num_objects), d_labels]
        tgt = torch.cat(targets)

        # Compute the L2 cost
        # [batch_size * num_queries, sum(num_objects)]
        assert probs.shape[0] == bs * num_queries
        assert probs.shape[1] == 1
        dist = torch.cdist(out, tgt, p=2)
        dist = dist.clamp_max(self.cutoff_distance)

        cost = torch.pow(input=dist, exponent=self.order) * probs
        cost += (1-probs) * (self.miss_cost) / 2.0

        assert cost.shape[0] == bs * num_queries
        assert cost.shape[1] == tgt.shape[0]

        # Clamp according to GOSPA
        # cost = cost.clamp_max(self.miss_cost)

        # Reshape
        # [batch_size, num_queries, sum(num_objects)]
        #cost = cost.view(bs, num_queries, -1)
        cost = cost.view(bs, num_queries, -1).cpu()

        # List with num_objects for each training-example
        sizes = [len(v) for v in targets]

        # Perform hungarian matching using scipy linear_sum_assignment
        with torch.no_grad():
            cost_split = cost.split(sizes, -1)
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_split)]
            
            permutation_idx = []
            unmatched_x = []
            for i, perm_idx in enumerate(indices):
                pred_idx, ground_truth_idx = perm_idx
                pred_unmatched = list(set(i for i in range(num_queries)) - set(pred_idx))

                permutation_idx.append((torch.as_tensor(pred_idx, dtype=torch.int64).to(self.device), torch.as_tensor(ground_truth_idx, dtype=torch.int64)))
                unmatched_x.append(torch.as_tensor(pred_unmatched, dtype=torch.int64).to(self.device))
               
        return permutation_idx, cost.to(self.device), unmatched_x
    
    def gospa_forward(self, outputs, targets, probabilistic=True, existance_threshold=0.75):

        assert 'state' in outputs, "'state' should be in dict"
        assert 'logits' in outputs, "'logits' should be in dict"

        output_state = outputs['state']
        output_logits = outputs['logits'].sigmoid()
        # List with num_objects for each training-example
        sizes = [len(v) for v in targets]

        bs = output_state.shape[0]
        if probabilistic:
            indices, cost_matrix, unmatched_x = self.compute_prob_gospa_matching(outputs, targets)
            cost_matrix = cost_matrix.split(sizes, -1)
            loss = 0
            for i in range(bs):
                batch_idx = indices[i]
                batch_cost = cost_matrix[i][i][batch_idx].sum()
                batch_cost = batch_cost + output_logits[i][unmatched_x[i]].sum() * self.miss_cost/2.0
                loss = loss + batch_cost
            loss = loss/sum(sizes)
            return loss, indices
        else:
            assert 0 < existance_threshold < 1, "'existance_threshold' should be in range (0,1)"
            loss, indices, decomposition = self.compute_orig_gospa_matching(outputs, targets, existance_threshold)
            loss = loss / bs
            return loss, indices, decomposition

    def state_loss(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src = outputs['state'][idx]
        target = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss = F.l1_loss(src, target)

        return loss

    def logits_loss(self, outputs, targets, indices):
        src_logits = outputs['logits']
        idx = self._get_src_permutation_idx(indices)

        '''
        can be modified to be used if classes are used 
        if isinstance(targets[0], dict):
            assert 'labels' in targets[0], "'labels' should be in dict"
            target_classes_o = torch.cat([t["labels"][i] for t, (_, i) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
            target_classes[idx] = target_classes_o
        '''
        
        target_classes = torch.zeros_like(src_logits, device=src_logits.device)
        target_classes[idx] = 1.0 # this is representation of an object
        loss = self.detr_logits_criterion(src_logits.squeeze(), target_classes.squeeze())

        return loss

    def get_loss(self, outputs, targets, loss_type, existance_threshold=None):
        if loss_type == 'gospa':
            loss, indices = self.gospa_forward(outputs, targets, probabilistic=True)
        elif loss_type == 'gospa_eval':
            loss,_ = self.gospa_forward(outputs, targets, probabilistic=False, existance_threshold=existance_threshold)
            indices = None
        elif loss_type == 'detr': 
            indices, _ = self.compute_hungarian_matching(outputs, targets)
            log_loss = self.logits_loss(outputs, targets, indices)
            state_loss = self.state_loss(outputs, targets, indices)
            loss = (state_loss + log_loss)
        
        return loss, indices
    
    def forward(self,outputs, targets, loss_type = 'detr', existance_threshold=0.75):

        assert 'state' in outputs, "'state' should be in dict"
        assert 'logits' in outputs, "'logits' should be in dict"

        outputs_without_aux = {k: v for k, v in outputs.items() if k not in ['aux_outputs', 'enc_outputs']}

        losses = {}

        if loss_type in ['gospa', 'gospa_eval', 'detr']:
            loss, indices = self.get_loss(outputs_without_aux, targets, loss_type, existance_threshold)
            losses.update({loss_type: loss}) 
        elif loss_type == 'both':
            for t in ['gospa', 'detr']:
                loss,indices = self.get_loss(outputs_without_aux, targets, t, existance_threshold)
                losses.update({t: loss}) 
        else:
            raise ValueError(f"The loss type should be either gospa, detr or both. Currently trying '{loss_type}'.'")

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if loss_type == 'both':
                    for t in ['gospa', 'detr']:
                        loss,_ = self.get_loss(aux_outputs, targets, t, existance_threshold)
                        losses.update({t+f'_{i}': loss}) 
                else:
                    aux_loss, _ = self.get_loss(aux_outputs, targets, loss_type, existance_threshold)
                    losses.update({loss_type+f'_{i}': aux_loss})

        if 'enc_outputs' in outputs:
            if loss_type == 'both':
                enc_outputs = outputs['enc_outputs']
                for t in ['gospa', 'detr']:
                    enc_loss, _ = self.get_loss(enc_outputs, targets, t, existance_threshold)
                    losses.update({t+'_enc': enc_loss})
            else:
                enc_outputs = outputs['enc_outputs']
                enc_loss, _ = self.get_loss(enc_outputs, targets, loss_type, existance_threshold)
                losses.update({loss_type+'_enc': enc_loss})

        return losses, indices

class FalseMeasurementLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device = torch.device(params.training.device)
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones([1])*params.data_generation.n_avg_false_measurements/params.data_generation.n_avg_starting_objects) # value >1 gives increased recall = less FN
        self.to(self.device)

    def forward(self, log_classifications, unique_ids):
        output = log_classifications.flatten()
        i = unique_ids.flatten()
        output = output[i != -2]
        i = i[i != -2]
        tgt = torch.zeros_like(output, device=output.device)
        tgt[i == -1] = 1

        return self.loss(output, tgt)/len(tgt)


class DataAssociationLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.device = torch.device(params.training.device)
        self.params = params
        self.sim = torch.nn.CosineSimilarity()
        # Use BCE or CE? i.e. allow one measurement to be from one or mutliple objects
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss(margin=0.5)
        self.to(self.device)
        

    def forward(self, first, second, missed_variable):
        first_embed = first['embed']
        first_ids = first['ids']

        second_embed = second['embed']
        second_ids = second['ids']

        bs = len(first_embed)

        loss = {}
        if self.params.loss.cosine_loss:
            loss['cosine'] = 0
        if self.params.loss.binary_cross_entropy_loss:
            loss['binary_cross_entropy'] = 0
        if self.params.loss.cross_entropy_loss:
            loss['cross_entropy'] = 0
        
        aff_matrix = []
        for i in range(bs):
            # filter out all false measurements
            true_objects = first_ids[i] != -1
            x = first_embed[i][true_objects]
            y = second_embed[i]
            if x.shape[0] == 0:
                continue

            # create target ids
            id_y = second_ids[i].cpu().numpy()
            id_x = first_ids[i][true_objects].cpu().numpy()
            missed_detection_embedding_idx = len(id_y)
            target = Tensor([np.where(id_y == s)[0][0] if (s in id_y) else missed_detection_embedding_idx for s in id_x]).to(self.device)


            if self.params.loss.cosine_loss:
                for (t, t_id) in zip(x,target):
                    cosine_target = -torch.ones((y.shape[0])).to(missed_variable.device)  
                    if t_id.long() != missed_detection_embedding_idx:          
                        cosine_target[t_id.long()] = 1

                    loss['cosine'] = loss['cosine'] + self.cosine_loss(t.unsqueeze(0), y, cosine_target)
                    

    
            # reshape to fit torch.nn.CosineSimilarity
            x = x.unsqueeze(dim=2)
            y = y.unsqueeze(dim=0).permute(0,2,1)
            # compute affinity matrix
            aff = self.sim(x, y)
            delta = torch.ones((x.shape[0],1)).to(missed_variable.device)*missed_variable
            aff = torch.cat((aff, delta), dim=1)
            aff_matrix.append(aff)

            
            # binaryCrossEntropy
            if self.params.loss.binary_cross_entropy_loss:
                target_one_hot = torch.nn.functional.one_hot(target.to(torch.int64), missed_detection_embedding_idx+1).type_as(aff)
                loss['binary_cross_entropy'] = loss['binary_cross_entropy'] + self.bce_loss(input=aff, target=target_one_hot)
            
            # crossEntropyLoss
            if self.params.loss.cross_entropy_loss:
                loss['cross_entropy'] = loss['cross_entropy'] + self.ce_loss(input=aff, target=target.long())
                
            loss = dict([(k, v/bs) for k,v in loss.items()])
        return loss, aff_matrix

            
           






    


        