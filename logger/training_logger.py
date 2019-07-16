
class TrainingLogger:
    """
    Logger, used by save training history
    """

    def __init__(self, dir_path,saving_loss_freq,saving_results_freq):
        self.dir_path = dir_path
        ensure_dir(self.dir_path)
        self.training_name = training_name
        self.saving_loss_freq=saving_loss_freq
        self.saving_results_freq=saving_results_freq

        path=os.path.join(self.dir_path, 'training_pickled/')
        ensure_dir(self.dir_path)
        self.results={'training_loss':[],
                    'test_loss':[],
                    'inputs':[]
                    'outputs'}

    def

