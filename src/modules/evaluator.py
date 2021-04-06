import torch
import time
import numpy as np
from util.misc import NestedTensor
#from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA


def compute_losses(outputs, labels, contrastive_classifications, unique_ids, mot_loss, contrastive_loss):
    c_loss = contrastive_loss.forward(contrastive_classifications, unique_ids)
    d_loss,_ = detr_loss.forward(outputs, labels, loss_type = 'detr')
    return d_loss, c_loss

def compute_gospa(outputs, labels, mot_loss, existance_threshold=0.9):
    loss, _, decomposition = mot_loss.gospa_forward(outputs, labels, False, existance_threshold)
    return loss, decomposition

def create_random_baseline(params):
    bs = params.training.batch_size
    nq = params.arch.num_queries
    d_det = params.arch.d_detections
    lb = params.data_generation.field_of_view_lb
    ub = params.data_generation.field_of_view_ub
    out = {}
    state = np.random.uniform(low=lb, high=ub, size=(bs, nq, d_det))
    logits = np.log(np.ones(shape=(bs,nq,1))) 

    out['state'] = torch.Tensor(state).to(torch.device(params.training.device))
    out['logits'] = torch.Tensor(logits).to(torch.device(params.training.device))

    return out

def create_all_measurement_baseline(batch, params):

    
    bs = params.training.batch_size
    nq = params.model.num_queries
    d_det = params.model.d_detections
    out = {}
    state = np.zeros(shape=(bs, nq, d_det))
    logits = np.log(np.ones(shape=(bs, nq, 1)) * 0.01)

    for i in range(bs):
        idx = batch[i,:,-1] == (params.general.n_timesteps - 1 - params.general.n_prediction_lag)*params.general.dt
        b = batch[i,idx] 
        num_meas = b.shape[0]
        state[i,:num_meas,:] = b[:,:d_det].cpu()
        logits[i,:num_meas,:] = np.log(0.99/(1-0.99))  

    out['state'] = torch.Tensor(state).to(torch.device(params.training.device))
    out['logits'] = torch.Tensor(logits).to(torch.device(params.training.device))

    return out

def create_true_measurement_baseline(batch, unique_ids, params):

    bs = params.training.batch_size
    nq = params.arch.num_queries
    d_det = params.arch.d_detections
    out = {}
    state = np.zeros(shape=(bs, nq, d_det))
    logits = np.log(np.ones(shape=(bs, nq, 1)) * 0.01)

    for i in range(bs):
        time_idx = batch[i,:,-1] == (params.data_generation.n_timesteps - 1 - params.data_generation.n_prediction_lag)*params.data_generation.dt
        true_idx = torch.logical_not(torch.logical_or(unique_ids[i,:]==-1, unique_ids[i,:]==-2))
        idx = torch.logical_and(time_idx, true_idx)
        b = batch[i,idx] 
        num_meas = b.shape[0]
        state[i,:num_meas,:] = b[:,:d_det].cpu()
        logits[i,:num_meas,:] = np.log(0.99/(1-0.99))  

    out['state'] = torch.Tensor(state).to(torch.device(params.training.device))
    out['logits'] = torch.Tensor(logits).to(torch.device(params.training.device))

    return out

def generate_tsne(embeddings, object_ids, ax):
    from sklearn.manifold import TSNE
    e = embeddings.detach().numpy()
    transformed_embeddings = TSNE(n_components=2,perplexity=50,n_iter=5000).fit_transform(e[object_ids!=-2])
    x = transformed_embeddings[:,0]
    y = transformed_embeddings[:,1]
    s = ax.scatter(x,y, c=(object_ids[object_ids!=-2]), cmap='tab20')
    l = ax.legend(*s.legend_elements(), title="IDs")
    ax.add_artist(l)

def generate_pca(embeddings, object_ids, ax):
    from sklearn.decomposition import PCA
    e = embeddings.detach().numpy()
    transformed_embeddings = PCA(n_components=2).fit_transform(e[object_ids!=-2])
    x = transformed_embeddings[:,0]
    y = transformed_embeddings[:,1]
    s = ax.scatter(x,y,c=(object_ids[object_ids!=-2]), cmap='tab20')
    l = ax.legend(*s.legend_elements(), title="IDs")
    ax.add_artist(l)


def generate_scene(embeddings, object_ids, ax):
    pass

def visualize_attn_maps(batch, outputs, attn_maps, ax, object_to_visualize=0, layer_to_visualize=-1):
    nq = attn_maps.shape[2]
    number_decoder_layers = attn_maps.shape[1]
    assert 0 <= object_to_visualize < nq, f"object to visualize should be in range of {0}-{nq}"
    if 'aux_outputs' in outputs and layer_to_visualize < number_decoder_layers - 1:
        outputs_state = outputs['aux_outputs'][layer_to_visualize]['state'].detach()
        outputs_prob = outputs['aux_outputs'][layer_to_visualize]['logits'].sigmoid().detach()
    else:
        outputs_state = outputs['state'].detach()
        outputs_prob = outputs['logits'].sigmoid().detach()

    attn_maps = attn_maps.detach()
    measurements = batch[0,:,:-1].numpy()
    attn_weights = attn_maps[0,layer_to_visualize,object_to_visualize, :].numpy()
    object_pos = outputs_state[0,object_to_visualize,:].numpy()
    object_prob = outputs_prob[0,object_to_visualize,:].numpy()

    label = "{:.2f}".format(object_prob[0])
    ax.annotate(label, # this is the text
                (object_pos[0], object_pos[1]), # this is the point to label
                textcoords="offset points", # how to position the text
                xytext=(0,10), # distance from text to points (x,y)
                ha='center',
                color='g')
    ax.plot(object_pos[0], object_pos[1], marker='o', color='g', label='Predicted object position', markersize=10)
    colors = np.zeros((measurements.shape[0], 4))
    colors[:, 3] = attn_weights/np.linalg.norm(attn_weights)
    ax.scatter(measurements[:,0], measurements[:,1], marker='x', color=colors, s=64)
    ax.legend()



def evaluate_metrics(data_generator, model, params, mot_loss, existance_threshold=0.75, num_eval=1000, verbose=False, data=None):
    orig_gospa = {'random': {'total': [], 'localization': [], 'localization_normalized': [], 'missed': [], 'false': []},
                  'output': {'total': [], 'localization': [], 'localization_normalized': [], 'missed': [], 'false': []},
                  'true_meas': {'total': [], 'localization': [], 'localization_normalized': [], 'missed': [], 'false': []}}
    prob_gospa = {'random': [], 'output': [], 'true_meas': []}
    detr = {'random': [], 'output': [], 'true_meas': []}

    for i in range(num_eval):
        t = time.time()
        if (i % (1+(num_eval // 10))) == 0:
            print(f"Starting to evaluate example {i+1}/{num_eval}....")
        if data_generator is None:
            batch, labels, unique_ids = data
        else:
            batch, labels, unique_ids, trajectories = data_generator.get_batch()
        output, memory, contrastive_classifications, queries, attn_maps = model.forward(batch, unique_ids)
        random = create_random_baseline(params)
        true_meas = create_true_measurement_baseline(batch.tensors, unique_ids, params)
        outs = [random, output, true_meas]
        for i, key in enumerate(['random', 'output', 'true_meas']):
            og, decomposition = compute_gospa(outs[i], labels, mot_loss, existance_threshold=params.loss.existence_prob_cutoff)
            pg,_ = mot_loss.gospa_forward(outs[i], labels)
            d,_ = mot_loss.forward(outs[i], labels, loss_type='detr')

            orig_gospa[key]['total'].append(og.item())
            orig_gospa[key]['localization'].append(decomposition['localization'])
            if decomposition['n_matched_objs'] != 0:
                orig_gospa[key]['localization_normalized'].append(decomposition['localization']/decomposition['n_matched_objs'])
            else:
                orig_gospa[key]['localization_normalized'].append(decomposition['localization'])
            orig_gospa[key]['missed'].append(decomposition['missed'])
            orig_gospa[key]['false'].append(decomposition['false'])
            detr[key].append(d['detr'].item())
            prob_gospa[key].append(pg.item())

            if verbose:
                print(f"'{key}' - Original GOSPA : {round(og.item(),4)}")
                print(f"'{key}' - Prob GOSPA: {round(pg.item(),4)}")
                print(f"'{key}' - DETR: {round(d['detr'].item(),4)}")
                
        if verbose:
            print(f"Evaluation took: {round(time.time() - t)} seconds...")
            print(".....................................................")

    return orig_gospa, prob_gospa, detr