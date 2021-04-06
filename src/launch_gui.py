import os

from data_generation.data_generator import DataGenerator
from util.misc import super_load
from util.load_config_files import load_yaml_into_dotdict
from modules.loss import MotLoss
from modules.contrastive_loss import ContrastiveLoss
from modules import evaluation_gui


base = ''
folder = "src/results/exp_foldername_here"
log_path = os.path.join(base, folder)
legacy = False
model, params = super_load(log_path, verbose=True)

eval_params = load_yaml_into_dotdict('configs/eval/default.yaml')
params.recursive_update(eval_params)


mot_loss = MotLoss(params)
contrastive_loss = ContrastiveLoss(params)

print('Launching GUI....')
data_generator = DataGenerator(params)
pmbm_results = 'pmbm_results.mat'
if os.path.isfile(pmbm_results):
    pmbm = evaluation_gui.PMBM(pmbm_results)
else:
    pmbm = None

app = evaluation_gui.MottGuiApp(model, data_generator, mot_loss, contrastive_loss, params, pmbm)
app.mainloop()
