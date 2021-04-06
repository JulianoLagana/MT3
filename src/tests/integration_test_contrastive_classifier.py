from data_generation.data_generator import DataGenerator
from util.load_config_files import dotdict
from util.plotting import contrastive_classifications_plot
from modules.loss import MotLoss
from modules.contrastive_loss import ContrastiveLoss
from modules.MOTT import MOTT
import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import argparse
import yaml
import os
import time
import datetime


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', '--filepath', help='filepath to configuration yaml file', required=True)
    args = parser.parse_args()

    if os.path.exists(args.filepath):
        with open(args.filepath, 'r') as f:
            try:
                params = yaml.safe_load(f)
                return params
            except yaml.YAMLError as exc:
                print(f"Error loading yaml file. Error: {exc}")
                exit()
    else:
        print(f"Filepath specified does not exist. Make sure '{args.fp}' is correct.")
        exit()


def convert_to_dot_dict(regular_dict):
    for key in regular_dict:
        if isinstance(regular_dict[key], dict):
            regular_dict[key] = convert_to_dot_dict(regular_dict[key])
    return dotdict(regular_dict)


if __name__ == '__main__':
    params = get_params()
    params = convert_to_dot_dict(params)
    model_params = params.model_params
    trainig_params = params.training
    debug_params = params.debug
    data_gen_params = params.data_generation_params

    model = MOTT(params)
    model.to(torch.device(params.general.device))
    data_generator = DataGenerator(params)
    mot_loss = MotLoss(params)
    contrastive_loss = ContrastiveLoss(params)

    optimizer = Adam(model.parameters(), lr=trainig_params.learning_rate)
    if debug_params.enable_plot:
        fig = plt.figure(constrained_layout=True, figsize=(15, 8))
        fig.canvas.set_window_title('Training Progress')

        gs = GridSpec(2, 3, figure=fig)
        loss_ax = fig.add_subplot(gs[0, 0])
        loss_ax.set_ylabel('Loss')
        loss_ax.grid('on')
        loss_line, = loss_ax.plot([1], 'r', label='Loss')

        percent_ax = fig.add_subplot(gs[1, 0])
        percent_ax.set_ylabel('Avg Certainty')
        percent_ax.grid('on')
        percent_line, = percent_ax.plot([1], 'r', label='Avg Certainty')
        percent_line_std_upper, = percent_ax.plot([1], 'b--', label='+3 \sigma certainty')
        percent_line_std_lower, = percent_ax.plot([1], 'b--', label='-3 \sigma certainty')
        percent_line_max, = percent_ax.plot([1], 'g--', label='+3 \sigma certainty')
        percent_line_min, = percent_ax.plot([1], 'g--', label='-3 \sigma certainty')

        output_ax = fig.add_subplot(gs[:, 1:])
        output_ax.set_ylabel('Y')
        output_ax.set_xlabel('X')

    losses = []
    avg_certainties = []
    std_certainties = []
    max_certainties = []
    min_certainties = []
    print("[INFO] Training started...")
    start_time = time.time()
    time_since = time.time()
    for epoch in range(trainig_params.num_epochs):
        for episode in range(trainig_params.num_episodes_per_epoch):
            try:
                batch, labels, unique_ids, trajectories = data_generator.get_batch(batch_size=trainig_params.batch_size)
                optimizer.zero_grad()
                outputs, memory, contrastive_classifications, queries, attn_maps = model.forward(batch, unique_ids)
                loss = contrastive_loss(contrastive_classifications, unique_ids)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                if episode % debug_params.print_interval == 0:
                    cur_time = time.time()
                    t = str(datetime.timedelta(seconds=round(cur_time - time_since)))
                    t_tot = str(datetime.timedelta(seconds=round(cur_time - start_time)))
                    print(
                        f"Epoch: {epoch + 1} \t Episode: {episode + 1} \t Loss: {np.mean(losses[-15:])} \t Time elasped: {t} \t Total time elapsed: {t_tot}")
                    time_since = time.time()

                if debug_params.enable_plot and episode % debug_params.plot_interval == 0:
                    x_axis = list(range(epoch * trainig_params.num_episodes_per_epoch + episode + 1))
                    loss_line.set_data(x_axis, losses)
                    loss_ax.relim()
                    loss_ax.autoscale_view()

                    output_ax.cla()
                    contrastive_classifications_plot(output_ax, batch, unique_ids, contrastive_classifications)
                    output_ax.relim()

                    fig.canvas.draw()
                    plt.pause(0.01)

            except KeyboardInterrupt:
                name = f'weights_epoch_{epoch}_episode_{episode}_{time.strftime("%Y%m%d_%H%M")}'
                print(f"[INFO] Saving weights in {name}")
                cur_path = os.path.dirname(os.path.abspath(__file__))
                folder_name = cur_path + os.sep + 'saved_models'
                if not os.path.isdir(folder_name):
                    os.makedirs(folder_name)

                torch.save(model.state_dict(), folder_name + os.sep + name)
                print("[INFO] Exiting...")
                exit()



