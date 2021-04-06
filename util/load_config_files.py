import collections.abc
import os
import yaml


def load_yaml_into_dotdict(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                params = yaml.safe_load(f)
                return convert_to_dot_dict(params)
            except yaml.YAMLError as exc:
                print(f"Error loading yaml file. Error: {exc}")
                exit()
    else:
        print(f"Filepath specified does not exist. Make sure '{filepath}' is correct.")
        exit()


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def recursive_update(self, u):
        """
        Recursive version of update() for dicts. If u has entries in common with self, overwrite. If an entry is a dict,
        go inside each key and call update() again.

        Example:
            v:   {
                  'a': 2,
                  'b': {
                          'x': 1,
                          'y': 2
                       }
                 }
            u: {
                  'c': 10,
                  'b': {
                          'x': 30,
                          'z': 40
                       }
                }
            after calling v.recursive_update(u):
            v:  {
                    'a': 2,
                    'c': 10,
                    'b': {
                            'x': 30,
                            'y': 2,
                            'z': 40
                         }
                }
        """
        return dotdict._recursive_update(self, u)

    @staticmethod
    def _recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = dotdict._recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d


def convert_to_dot_dict(regular_dict):
    for key in regular_dict:
        if isinstance(regular_dict[key], dict):
            regular_dict[key] = convert_to_dot_dict(regular_dict[key])
    return dotdict(regular_dict)
