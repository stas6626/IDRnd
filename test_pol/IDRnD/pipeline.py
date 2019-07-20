import numpy as np
import torch
import pandas as pd


class Train:
    def __init__(self, callbacks=[]):
        self.callbacks = callbacks
        self.acumulate_factor = 1

    def fit(self, train_loader, val_loader, model, criterion, optimizer, epoches=100):

        for callback in self.callbacks:
            callback.on_train_begin()

        tr_cnt, val_cnt = 0, 0
        for epoch in range(epoches):
            model.train()

            for callback in self.callbacks:
                callback.on_epoch_begin(epoch=epoch, pipeline=self)

            for x_batch, y_batch in train_loader:
                preds = model(x_batch.float().cuda())

                loss = criterion(preds, y_batch.cuda())
                loss.backward()
                if tr_cnt % self.acumulate_factor == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                for callback in self.callbacks:
                    callback.on_train_batch_end(loss=loss.item(), iteration=tr_cnt)
                tr_cnt += 1

            if val_loader is not None:

                model.eval()
                valid_preds = np.zeros((len(val_loader.dataset), 1))

                for i, (x_batch, y_batch) in enumerate(val_loader):
                    with torch.no_grad():
                        preds = model(x_batch.float().cuda()).detach()
                        loss = criterion(preds, y_batch.cuda())

                    valid_preds[
                        i * val_loader.batch_size : (i + 1) * val_loader.batch_size
                    ] = preds.cpu().numpy()

                    val_cnt += 1
                    for callback in self.callbacks:
                        callback.on_val_batch_end(loss=loss.item(), iteration=val_cnt)

            for callback in self.callbacks:
                callback.on_epoch_end(
                    epoch=epoch,
                    y_true=val_loader.dataset.y,
                    y_pred=valid_preds.reshape(-1),
                    model=model.module,
                )

    def predict_on_test(self, test_loader, model):
        all_outputs, all_fnames = [], []
        model.eval()
        with torch.no_grad():
            for images, fnames in test_loader:
                preds = model(images.float().cuda()).detach()
                all_outputs.append(preds.cpu().numpy())
                all_fnames.extend(fnames)

        test_preds = pd.DataFrame(data=np.concatenate(all_outputs), index=all_fnames)
        # test_preds = test_preds.groupby(level=0).mean()
        return test_preds