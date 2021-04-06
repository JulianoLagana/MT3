import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, order, cutoff_distance, alpha):
        super().__init__()
        self.order = order
        self.cutoff_distance = cutoff_distance
        self.alpha = alpha

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: Tensor of dim [batch_size, num_queries, d_label] 

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
        bs, num_queries = outputs.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out = outputs.flatten(0, 1)                 # [batch_size * num_queries, d_label]
    
        # Also concat the target labels 
        tgt = torch.cat(targets)                    # [sum(num_objects), d_labels]
    
        # Compute the L2 cost 
        cost = torch.cdist(out, tgt, p=2) ** self.p          # [batch_size * num_queries, sum(num_objects)]
        
        # Clamp according to Gospa
        cost = cost.clamp_max(self.cutoff_distance ** self.order) 

        # Reshape
        cost = cost.view(bs, num_queries, -1).cpu() # [batch_size, num_queries, sum(num_objects)]

        # List with num_objects for each training-example
        sizes = [len(v) for v in targets]

        # Perform hungarian matching using scipy linear_sum_assignment
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
