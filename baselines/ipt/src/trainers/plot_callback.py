import lightning as L

class PlotCallback(L.Callback):
    """
    Plots reconstructions every N epochs. 
    Set plot_every_n_epochs=-1 to disable.
    """

    def __init__(self, every_n_epochs: int, plot_dir: str):
        self.every_n_epochs = every_n_epochs
        self.plot_dir = plot_dir

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.every_n_epochs == -1:
            return
        if trainer.current_epoch % self.every_n_epochs == 0:
            pl_module.save_plots(self.plot_dir, trainer.current_epoch)