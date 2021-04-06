"""tkinter app for evaluation GUI. Run evaluation.py to try it out!
"""
import matplotlib
import os
import torch
from pathlib import Path
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.io import loadmat

import tkinter as tk
from tkinter import ttk
import modules.evaluator as evaluator
import util.plotting as plotting
from util.misc import inverse_sigmoid

plot_options = [(evaluator.generate_tsne, 0, "Plot TSNE"),
                (evaluator.generate_pca, 1, 'Plot PCA'),
                (plotting.output_truth_plot_for_fusion_paper, 2, 'Plot scene'),
                (evaluator.visualize_attn_maps, 3, "Plot attention")]

class PMBM():
    def __init__(self, mat_file):
        if os.path.isfile(mat_file) and mat_file.endswith('.mat'):
            self.mat_data = mat_data = loadmat(mat_file)
        else:
            raise Exception('Must provide valid matfile.')

    def forward(self, i_sample):
        state = torch.Tensor(self.mat_data['est'][i_sample][0].T).unsqueeze(0)
        if 'est_existence_prob' in self.mat_data:
            logits = inverse_sigmoid(torch.Tensor(self.mat_data['est_existence_prob'][i_sample][0]).unsqueeze(-1))
        else:
            logits = torch.ones((1, state.shape[1], 1)) * 5
        outputs = {'state': state, 'logits': logits}
        return outputs


class MottGuiApp(tk.Tk):
    def __init__(self, model, data_generator, mot_loss, contrastive_loss, params, pmbm=None):
        # Init super
        tk.Tk.__init__(self)

        # Title
        self.title("Greatest GUI of all time")

        # Icon
        dirname = Path(os.path.dirname(__file__))
        img = tk.Image("photo", file=dirname.parent.parent / "docs" / "optimus_prime.png")
        self.tk.call('wm','iconphoto', self._w, img)

        # MOTT stuff: model, data_generator, losses and associated params
        self.model = model
        self.data_generator = data_generator
        self.mot_loss = mot_loss
        self.contrastive_loss = contrastive_loss
        self.params = params
        self.pmbm = pmbm

        # Initialize list with training data and get first example
        self.training_data = []
        self.model_outputs = []
        self.model_losses = []
        self.get_new_training_data()
        self.training_data_index = 0
        self.object_query_index = 0
        self.object_query_layer = params.arch.decoder.n_layers - 1

        # Catch closing of window
        self.protocol("WM_DELETE_WINDOW", self.close_window)

        # Start figure and plot dummy values
        self.figure = Figure(figsize=(5,5), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.grid('on')
        self.axis.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        # Add figure to GUI
        canvas_row_span = 20
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, rowspan=canvas_row_span, sticky=tk.N+tk.S+tk.E+tk.W)
        tk.Grid.rowconfigure(self, 0, weight=1)
        tk.Grid.columnconfigure(self, 0, weight=1)

        toolbar_frame = tk.Frame(master=self)
        toolbar_frame.grid(row=canvas_row_span+1, column=0, columnspan=3, sticky=tk.W)
        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        toolbar.update()

        # Radio buttons for selecting type of plot
        self.rb_var = tk.IntVar()
        self.radio_buttons = []
        for plot_type, val, text in plot_options:
            radio_button = tk.Radiobutton(self, text=text, variable=self.rb_var, command=self.update_plot, value=val)
            radio_button.grid(row=canvas_row_span-val, column=5, sticky=tk.N+tk.E+tk.W)
            self.radio_buttons.append(radio_button)
        
        # Buttons for changing training example
        self.button_prev = tk.Button(self, text='<-', command=self.previous)
        self.button_prev.grid(row=canvas_row_span+1, column=3)
        self.button_prev['state'] = 'disabled'

        self.button_next = tk.Button(self, text='->', command=self.next)
        self.button_next.grid(row=canvas_row_span+1, column=4)

        self.object_button_next = tk.Button(self, text='↑ obj', command=self.object_next)
        self.object_button_next.grid(row=1, column=5)
        self.object_button_next['state'] = 'disabled'

        self.object_button_prev = tk.Button(self, text='↓ obj', command=self.object_previous)
        self.object_button_prev.grid(row=2, column=5)
        self.object_button_prev['state'] = 'disabled'

        self.layer_button_next = tk.Button(self, text='↑ layer', command=self.layer_next)
        self.layer_button_next.grid(row=3, column=5)
        self.layer_button_next['state'] = 'disabled'

        self.layer_button_prev = tk.Button(self, text='↓ layer', command=self.layer_previous)
        self.layer_button_prev.grid(row=4, column=5)
        self.layer_button_prev['state'] = 'disabled'

        # Label showing training example count and current training example
        self.label_training_example = tk.Label(self, text="1/1")
        self.label_training_example.grid(row=canvas_row_span+1, column=5)

        # Labels for losses in current scene
        self.label_og_gospa_loss = [tk.Label(self, text="OG GOSPA").grid(row=2, column=3)]
        for i in range(3):
            lbl = tk.Label(self, text="")
            lbl.grid(row=i+1, column=4)
            self.label_og_gospa_loss.append(lbl)

        self.label_pg_gospa_loss = [tk.Label(self, text="PG GOSPA").grid(row=6, column=3)]
        for i in range(3):
            lbl = tk.Label(self, text="")
            lbl.grid(row=i+5, column=4)
            self.label_pg_gospa_loss.append(lbl)

        self.label_detr_loss = [tk.Label(self, text="DETR").grid(row=10, column=3)]
        for i in range(3):
            lbl = tk.Label(self, text="")
            lbl.grid(row=i+9, column=4)
            self.label_detr_loss.append(lbl)

        # Update figure to show relevant plot
        self.update_plot()
        self.update_loss_label()

        
    def update_plot(self):
        """Updates plot according to training example and selected plot type
        """
        matplotlib.rcParams['mathtext.fontset'] = 'cm'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        plot_option = self.rb_var.get()
        batch, labels, unique_ids = self.training_data[self.training_data_index]
        self.axis.clear()
        
        # Check if we've computed outputs before
        if self.model_outputs[self.training_data_index] == None:
            outputs, memory, contrastive_classifications, queries, attn_maps = self.model.forward(batch, unique_ids)
            #self.model_outputs[self.training_data_index] = (outputs, memory, contrastive_classifications, queries, attn_maps)
        else:
            outputs, memory, contrastive_classifications, queries, attn_maps = self.model_outputs[self.training_data_index]

        self.object_button_prev['state'] = 'disabled'
        self.object_button_next['state'] = 'disabled'
        self.layer_button_prev['state'] = 'disabled'
        self.layer_button_next['state'] = 'disabled'

        if plot_option == 0:
            plot_options[0][0](memory[0], unique_ids[0], self.axis)

        elif plot_option == 1:
            plot_options[1][0](memory[0], unique_ids[0], self.axis)

        elif plot_option == 2:
            loss, indices = self.mot_loss.forward(outputs, labels, loss_type=self.params.loss.type)
            #plot_options[2][0](self.axis, outputs, labels, indices, batch)
            plot_options[2][0](self.axis, outputs, labels, batch, unique_ids, self.params)

            if self.pmbm is not None:
                pmbm_output = self.pmbm.forward(self.training_data_index)
                pmbm_out = pmbm_output['state'][0]
                pmbm_logits = pmbm_output['logits'][0][:, 0]
                filtered_pmbm_out = pmbm_out[pmbm_logits > 0.5]
                self.axis.scatter(filtered_pmbm_out[:, 0], filtered_pmbm_out[:, 1], s=200, marker='x', label='PMBM predictions', c='g')
            
            self.axis.set_xlim([self.params.data_generation.field_of_view_lb, self.params.data_generation.field_of_view_ub])
            self.axis.set_ylim([self.params.data_generation.field_of_view_lb, self.params.data_generation.field_of_view_ub])
            self.axis.set_xlabel('x (m)', fontdict={'family': 'STIXGeneral'})
            self.axis.set_ylabel('y (m)', fontdict={'family': 'STIXGeneral'})
            self.axis.yaxis.labelpad = -10

            self.axis.figure.savefig('test.pdf', bbox_inches='tight')

        elif plot_option == 3: 
            plot_options[3][0](batch.tensors, outputs, attn_maps, self.axis, object_to_visualize=self.object_query_index, layer_to_visualize=self.object_query_layer)
            self.axis.set_xlim([self.params.data_generation.field_of_view_lb, self.params.data_generation.field_of_view_ub])
            self.axis.set_ylim([self.params.data_generation.field_of_view_lb, self.params.data_generation.field_of_view_ub])
            self.set_attention_button_states()
            
        self.canvas.draw()

    def get_new_training_data(self):
        """Creates new training examples and saves them to memory as needed
        """
        batch, labels, unique_ids, trajectories = self.data_generator.get_batch()
        self.training_data.append((batch,labels,unique_ids))
        self.model_outputs.append(None)
        self.model_losses.append(None)

    def update_training_example_label(self):
        """Updates the label, showing which training example is being examined.
        """
        self.label_training_example['text'] = f"{self.training_data_index+1}/{len(self.training_data)}"

    def update_loss_label(self):
        if self.model_losses[self.training_data_index] == None:
            data = self.training_data[self.training_data_index]
            params = self.params
            params.training.batch_size = 1
            og, pg, d = evaluator.evaluate_metrics(None, self.model, params, self.mot_loss,  num_eval=1, verbose=False, data=data)
            self.model_losses[self.training_data_index] = (og, pg, d)
        else:
            og, pg, d = self.model_losses[self.training_data_index]

        for i, (og_key, pg_key, d_key) in enumerate(zip(og,pg,d)):
            self.label_og_gospa_loss[i+1]['text'] = "{}: {:.2f}".format(og_key, og[og_key]['total'][0])
            self.label_pg_gospa_loss[i+1]['text'] = "{}: {:.2f}".format(pg_key, pg[pg_key][0])
            self.label_detr_loss[i+1]['text'] = "{}: {:.2f}".format(d_key, d[d_key][0])
        
    
    def set_attention_button_states(self):
        if self.object_query_index == 0:
            self.object_button_prev['state'] = 'disabled'
        else:
            self.object_button_prev['state'] = 'normal'
        
        if self.object_query_index == self.params.arch.num_queries-1:
            self.object_button_next['state'] = 'disabled'
        else:
            self.object_button_next['state'] = 'normal'
        
        if self.object_query_layer == self.params.arch.decoder.n_layers-1:
            self.layer_button_next['state'] = 'disabled'
        else:
            self.layer_button_next['state'] = 'normal'
        
        if self.object_query_layer == 0:
            self.layer_button_prev['state'] = 'disabled'
        else:
            self.layer_button_prev['state'] = 'normal'


    def object_previous(self):
        """Handle button press for "previous attention object"
        """
        self.object_query_index -= 1 
        self.update_plot()
        self.update_training_example_label()
        self.update_loss_label()

    def object_next(self):
        """Handle button press for "next attention object"
        """
        self.object_query_index += 1
        self.update_plot()
        self.update_training_example_label()
        self.update_loss_label()

    def layer_previous(self):
        """Handle button press for "previous attention layer"
        """
        self.object_query_layer -= 1
        self.update_plot()
        self.update_training_example_label()
        self.update_loss_label()


    def layer_next(self):
        """Handle button press for "next attention layer"
        """
        self.object_query_layer += 1
        self.update_plot()
        self.update_training_example_label()
        self.update_loss_label()


    def previous(self):
        """Handle button press for "previous training example"
        """
        if self.training_data_index == 0:
            return
        else:
            self.training_data_index -= 1
        
        if self.training_data_index == 0:
            self.button_prev['state'] = 'disabled'

        self.update_plot()
        self.update_training_example_label()
        self.update_loss_label()

    def next(self):
        """Handle button press for "next training example"
        """
        if self.training_data_index == len(self.training_data) - 1:
            self.get_new_training_data()
        
        self.training_data_index += 1
        self.button_prev['state'] = 'normal'
        self.update_plot()
        self.update_training_example_label()
        self.update_loss_label()

    def finish(self):
        self.close_window()

    def close_window(self):
        """Catch window closing
        """
        print('Goodbye have a nice day')
        self.destroy()
