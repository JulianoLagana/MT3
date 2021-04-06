import yaml
from util.load_config_files import dotdict

standard_values = {
    'n_timesteps':  10,
    'max_objects':  32, 
    'device':       'cpu',
    'dt':           0.1,       
    'p_add':        0.4,    
    'p_remove':     0.05,
    'p_meas':       0.9,
    'sigma_q':      1,
    'sigma_y':      0.1,
    'n_avg_false_measurments' : 1,
    'n_avg_starting_objects': 4, 
    'false_measure_lb': -10,
    'false_measure_ub': 10,
    'mu_x0':        [0, 0],
    'std_x0':       [[3, 0], [0, 3]],
    'mu_v0':        [0, 0],
    'std_v0':       [[3, 0], [0, 3]]
}

        
class Configurator:
    def __init__(self,yaml_fp):
        self.fp = yaml_fp
        print(f"[CONFIG] Loading configuration from '{self.fp}'")
        with open(self.fp, 'r') as stream:
            self.input = yaml.safe_load(stream)
        self.args = dotdict()

    def configurate_key(self, key):
        value = self.input.get(key)
        if value is None:
            value = standard_values.get(key)
            print(f"[CONFIG] Can not find '{key}' in the YAML-file. Using standard value for {key} = {value}")
        
        self.args[key] = value

    def configurate(self):
        print('[CONFIG] STARTING CONFIGURATION...')
        for key in standard_values:
            self.configurate_key(key)
        
        print('[CONFIG] CONFIGURATION FINISHED SUCCESSFULLY...')
        return self.args

