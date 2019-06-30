try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("No tensorboard")
import torch


class Callback(object):
    def __init__(self):
        pass

    def on_train_batch_begin(self, *args, **params):
        pass

    def on_train_batch_end(self, *args, **params):
        pass

    def on_val_batch_begin(self, *args, **params):
        pass

    def on_val_batch_end(self, *args, **params):
        pass

    def on_epoch_begin(self, *args, **params):
        pass

    def on_epoch_end(self, *args, **params):
        pass

    def on_train_begin(self, *args, **params):
        pass

    def on_train_end(self, *args, **params):
        pass


class TensorBoardCallback(Callback):
    def __init__(self, scorer=None):
        self.scorer = scorer

    def on_train_begin(self):
        self.writer = SummaryWriter()

    def on_train_batch_end(self, **data):
        self.writer.add_scalar("train_loss", data["loss"], data["iteration"])

    def on_val_batch_end(self, **data):
        self.writer.add_scalar("val_loss", data["loss"], data["iteration"])

    def on_epoch_end(self, **data):
        if self.scorer is None:
            raise ("No scoring function")
        score = self.scorer(data["y_true"], data["y_pred"])
        self.writer.add_scalar("val_error", score, data["epoch"])


class SaveEveryEpoch(Callback):
    def __init__(self, model_path):
        self.model_path = model_path

    def on_epoch_end(self, **data):
        torch.save(data["model"].state_dict(), self.model_path + str(data["epoch"]))


class AccumulateGradient(Callback):
    def __init__(self, gradient_acumulation=[5, 10, 15, 20, 30, 40]):
        self.gradient_acumulation = gradient_acumulation

    def on_epoch_begin(self, **data):
        if data["epoch"] in self.gradient_acumulation:
            data["pipeline"].acumulate_factor *= 2


class SaveBestEpoch(Callback):
    def __init__(self, model_path):
        self.model_path = model_path
        self.best_score = 1

    def on_epoch_end(self, **data):
        if self.scorer is None:
            raise ("No scoring function")
        score = self.scorer(data["y_true"], data["y_pred"])
        if self.best_score > score:
            torch.save(data["model"].state_dict(), self.model_path)
            self.best_score = score


class EpochScheduler(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_begin(self, **data):
        self.sum_train_loss = 0.0

    def on_train_batch_end(self, **data):
        self.sum_train_loss += data["loss"]

    def on_epoch_end(self, **data):
        self.scheduler.step(self.sum_train_loss)