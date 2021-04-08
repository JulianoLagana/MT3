import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt


@torch.no_grad()
def output_truth_plot(ax, output, labels, matched_idx, batch, training_example_to_plot=0):
        assert 'state' in output, "'state' should be in dict"
        assert 'logits' in output, "'logits' should be in dict"
    
        output_state = output['state']
        output_logits = output['logits']
        bs, num_queries = output_state.shape[:2]
        assert training_example_to_plot <= bs, "'training_example_to_plot' should be less than batch_size"

        truth = labels[training_example_to_plot].cpu().numpy()
        indicies = tuple([t.cpu().detach().numpy() for t in matched_idx[training_example_to_plot]])
        out = output_state[training_example_to_plot].cpu().detach().numpy()
        out_prob = output_logits[training_example_to_plot].cpu().sigmoid().detach().numpy().flatten()
        for i in range(len(out)):
            if i in indicies[0]:
                tmp_idx = np.where(indicies[training_example_to_plot] == i)[0][0]
                truth_idx = indicies[1][tmp_idx]
                p = ax.plot(out[i, 0], out[i, 1], marker='o', label='Matched Predicted Object', markersize=10)
                ax.plot(truth[truth_idx, 0], truth[truth_idx, 1], marker='D', color=p[0].get_color(), label='Matched Predicted Object', markersize=10)
            else:
                p = ax.plot(out[i,0], out[i,1], marker='*', color='k', label='Unmatched Predicted Object', markersize=5)

            label = "{:.2f}".format(out_prob[i])
            ax.annotate(label, # this is the text
                        (out[i,0], out[i,1]), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center',
                        color=p[0].get_color())

            # Plot measurements, alpha-coded by time
            measurements = batch.tensors[training_example_to_plot][~batch.mask[training_example_to_plot]]
            colors = np.zeros((measurements.shape[0], 4))
            unique_time_values = np.array(sorted(list(set(measurements[:, 2].tolist()))))
            def f(t):
                """Exponential decay for alpha in time"""
                idx = (np.abs(unique_time_values - t)).argmin()
                return 1/1.5**(len(unique_time_values)-idx)
            colors[:, 3] = [f(t) for t in measurements[:, 2].tolist()]
            ax.scatter(measurements[:, 0].cpu(), measurements[:, 1].cpu(), marker='x', c=colors, zorder=np.inf)


@torch.no_grad()
def output_truth_plot_for_fusion_paper(ax, output, labels, batch, unique_idxs, params, training_example_to_plot=0):
    assert 'state' in output, "'state' should be in dict"
    assert 'logits' in output, "'logits' should be in dict"

    output_state = output['state']
    output_logits = output['logits']
    bs, num_queries = output_state.shape[:2]
    assert training_example_to_plot <= bs, "'training_example_to_plot' should be less than batch_size"

    truth = labels[training_example_to_plot].cpu().numpy()
    out = output_state[training_example_to_plot].cpu().detach().numpy()
    out_prob = output_logits[training_example_to_plot].cpu().sigmoid().detach().numpy().flatten()

    # Plot MT3 predictions
    for i in range(len(out)):
        if out_prob[i] >= params.loss.existence_prob_cutoff:
            p = ax.scatter(out[i, 0], out[i, 1], marker='+', s=200, c='b')

    # Plot ground-truth
    ax.scatter(truth[:, 0], truth[:, 1], marker='o', s=25, c='r', zorder=np.inf)

    # Plot measurements, alpha-coded by time
    measurements = batch.tensors[training_example_to_plot][unique_idxs[training_example_to_plot] != -1]
    colors = np.zeros((measurements.shape[0], 4))
    unique_time_values = np.array(sorted(list(set(measurements[:, 2].tolist()))))
    def f(t):
        """Exponential decay for alpha in time"""
        idx = (np.abs(unique_time_values - t)).argmin()
        return 1 / 1.1 ** (len(unique_time_values) - idx)
    colors[:, 3] = [f(t) for t in measurements[:, 2].tolist()]
    ax.scatter(measurements[:, 0].cpu(), measurements[:, 1].cpu(), marker='x', c=colors, zorder=-np.inf)

    # Plot false measurements, alpha-coded by time
    measurements = batch.tensors[training_example_to_plot][unique_idxs[training_example_to_plot] == -1]
    colors = np.zeros((measurements.shape[0], 4))
    unique_time_values = np.array(sorted(list(set(measurements[:, 2].tolist()))))
    colors[:, 3] = [f(t) for t in measurements[:, 2].tolist()]
    ax.scatter(measurements[:, 0].cpu(), measurements[:, 1].cpu(), marker='.', c=colors, zorder=-np.inf, s=10)


@torch.no_grad()
def contrastive_classifications_plot(ax, batch, object_ids, contrastive_classifications):

    measurements = batch.tensors[0]
    n_measurements = measurements.shape[0]
    classifications = contrastive_classifications[0]
    is_there_pads = np.min(object_ids[0].numpy()) == -2
    if is_there_pads:
        n_measurements_to_use = np.argmin(object_ids[0]).item()
    else:
        n_measurements_to_use = n_measurements
    object_ids = object_ids[0][:n_measurements_to_use].int().tolist()

    # Choose random measurement (not including padding measurements)
    chosen_measurement_idx = np.random.choice(n_measurements_to_use)
    chosen_object_id = int(object_ids[chosen_measurement_idx])

    # Assign a different color (if possible) to each of the objects in the scene, but the chosen object is always blue
    # and false measurements always red
    available_colors = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
                        'tab:cyan', 'sandybrown', 'goldenrod', 'lime', 'cyan']
    unique_ids = set(object_ids)
    color_dict = {i: available_colors[i % len(available_colors)] for i in unique_ids if i not in [chosen_object_id, -1]}
    color_dict[chosen_object_id] = 'tab:blue'
    color_dict[-1] = 'tab:red'
    bar_colors = list(map(color_dict.get, object_ids))

    # Plot color-coded predicted pmf for the chosen measurement
    chosen_classifications = classifications[chosen_measurement_idx].exp().detach()
    ax.bar(range(n_measurements_to_use), chosen_classifications.numpy()[:n_measurements_to_use], color=bar_colors)


@torch.no_grad()
def compute_avg_certainty(outputs_history, matched_idx_history):
    matched_certainties = []
    unmatched_certainties = []
    for outputs, matched_idx in zip(outputs_history, matched_idx_history):
        idx = _get_src_permutation_idx(matched_idx)
        output_logits = outputs['logits']

        mask = torch.zeros_like(output_logits).bool()
        mask[idx] = True
        matched_certainties.extend(output_logits[mask].sigmoid().cpu().tolist())
        unmatched_certainties.extend(output_logits[~mask].sigmoid().cpu().tolist())

    if len(matched_certainties) > 0:
        matched_quants = np.quantile(matched_certainties, [0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        matched_quants = [-1, -1, -1, -1, -1]

    if len(unmatched_certainties) > 0:
        unmatched_quants = np.quantile(unmatched_certainties, [0.0, 0.25, 0.5, 0.75, 1.0])
    else:
        unmatched_quants = [-1, -1, -1, -1, -1]

    return tuple(matched_quants), tuple(unmatched_quants)


def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i)
                        for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx
        

def get_constrastive_ax():
    fig, ax = plt.subplots()
    ax.grid('on')
    line, = ax.plot([1], 'r', label='Contrastive loss')
    ax.set_ylabel('Contrastive loss')

    return fig, ax, line

def get_false_ax():
    fig, ax = plt.subplots()
    ax.grid('on')
    line, = ax.plot([1], 'r', label='False loss')
    ax.set_ylabel('False loss')

    return fig, ax, line

def get_total_loss_ax():
    fig, ax = plt.subplots()
    ax.grid('on')
    line, = ax.plot([1], 'r', label='Total loss')
    ax.set_ylabel('Total loss')
    ax.set_yscale('log')
    
    return fig, ax, line


def get_new_ax(log=False, ylabel='Loss'):
    fig, ax = plt.subplots()
    ax.grid('on')
    line, = ax.plot([1], 'r', label='Loss')
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale('log')

    return fig, ax, line
    
