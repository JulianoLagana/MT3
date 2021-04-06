import os
import shutil
import sys


class FileAndStream(object):
    """Class used for redirecting output of a script to a file, but also displaying it in the terminal."""
    def __init__(self, filename, stream):
        self.terminal = stream
        self.logfile = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # Needed for Python 3 compatibility
        self.logfile.flush()
        self.terminal.flush()


class Logger:

    def __init__(self, log_path, save_output=True):
        """
        Constructor for the Logger class.

        Args:
            log_path:    Desired location for saving the experiment. If this location already exists, Logger will
                         append an ' (i)' to the end of this path, where i is the first integer greater than 1 that
                         makes this path be new.
            save_output: If True everything printed to the console will be saved in the files out.log and out_err.log.
        """
        # Append a number to the end of the desired path if that path already exists
        new_log_path = log_path
        i = 2
        while os.path.isdir(new_log_path):
            new_log_path = log_path + ' ({})'.format(i)
            i += 1

        self.log_path = new_log_path
        os.makedirs(new_log_path)

        # If desired, save all output to log files
        if save_output:
            sys.stdout = FileAndStream(self.log_path + '/out.log', sys.stdout)
            sys.stderr = FileAndStream(self.log_path + '/out_err.log', sys.stderr)

    def log_scalar(self, name, value, t):
        """
        Logs the given value into the file self.log_path/name.txt

        Args:
            name:  Name that will be logged.
            value: Value to be logged.
            t:     Time (or batch number, or optimization step) when this value was obtained. Usually used as the x-axis
                   when later plotting the logged value.
        """

        filename = self.log_path + '/' + name + '.txt'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a') as f:
            f.write('{}: {}\n'.format(t, value))

    def save_code_dependencies(self, project_root_path=None, additional_paths=None):
        """
        Copies the files of all imported modules present inside the folder `project_root_path` to the log directory, in
        order to guarantee reproducibility of the experiment.

        For most use cases, this method should be called when you're sure that all the modules you care about have
        already been imported by your script, so that all of them are copied to the log directory.

        Args:
            project_root_path: Folder that contains the modules that should be copied if imported.
            additional_paths: List of additional folders from which imported files should be copied.
        """

        # If project_root_path is not specified, assume it to be the path of the script being run
        if project_root_path is None:
            project_root_path = os.path.realpath(sys.path[0])

        if additional_paths is None:
            additional_paths = []

        # Iterate over all imported modules
        modules = sys.modules.values()
        for module in modules:
            if hasattr(module, '__file__'):
                filename = module.__file__
                if filename is not None:
                    # Check if imported module is inside any of the additional_paths
                    file_in_additional_paths = False
                    for p in additional_paths:
                        if p in filename:
                            file_in_additional_paths = True

                    # Save the current file if it is inside the project root folder, if it is the name of the script
                    # being run, or if it is inside any of the folders specified in additional_paths
                    if project_root_path in filename or filename == sys.argv[0] or file_in_additional_paths:
                        destination = self.log_path + '/code_used/' + filename.replace(project_root_path, '')
                        destination_folder = destination[:destination.rfind('/')]
                        if not os.path.exists(destination_folder):
                            os.makedirs(destination_folder)
                        shutil.copyfile(filename, destination)


def load_data(log_folder, var_name):
    """
    Function used to load data saved by Logger.log_scalar

    Args:
        log_folder: Name of the log folder where Logger saved the variable.
        var_name: Name of the variable that should be loaded.

    Returns:
        xs: list of all the x-values loaded.
        ys: list of all the y-values loaded.
    """
    log_folder = f'{log_folder}'
    filename = f'{var_name}.txt'

    # Safe open, with helpful error messages
    try:
        f = open(f'{log_folder}/{filename}', 'r')
    except FileNotFoundError as fne:
        if not os.path.isdir(log_folder):
            raise NotADirectoryError(f'Could not find directory {log_folder}')
        elif not os.path.isfile(filename):
            raise FileNotFoundError(f'Could not find file {filename} inside {log_folder}')
        else:
            raise fne

    xs, ys = [], []
    for line in f:
        l = line.split(':')
        xs.append(float(l[0]))
        ys.append(float(l[1]))
    return [xs, ys]
