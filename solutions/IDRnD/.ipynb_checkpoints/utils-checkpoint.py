import numpy as np
import torch
import os
import gc
import random
import pandas as pd
import time
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("No tensorboard")


from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


class Train:
    def __init__(self, model_path='weight_best.pt',  gradient_acumulation=[]):
        self.model_path = model_path
        self.gradient_acumulation = gradient_acumulation

    def fit(self, train_loader, val_loader, model, criterion, optimizer, scheduler, epoches=100):
        writer = SummaryWriter()
        best_epoch = -1
        best_err = 1
        acumulate_factor = 1
        tr_cnt, val_cnt = 0, 0  ## TODO
        for epoch in range(epoches):
            model.train()
            if epoch in self.gradient_acumulation:
                acumulate_factor *= 2
            avg_train_loss = 0.
            for x_batch, y_batch in train_loader:
                preds = model(x_batch.float().cuda())
                loss = criterion(preds, y_batch.cuda())
                loss.backward()
                if torch.isnan(loss): print(loss.item())
                if tr_cnt % acumulate_factor == 0:
                    optimizer.step()
                    #if scheduler is not None: scheduler.step();  writer.add_scalar("lr", scheduler.get_lr()[0], tr_cnt)
                    optimizer.zero_grad()

                tr_cnt += 1
                writer.add_scalar("train_loss", loss.item(), tr_cnt)
                avg_train_loss+=loss.item()
            if scheduler is not None: scheduler.step(avg_train_loss); tr_cnt

            if val_loader is None: continue
            gc.collect()
            model.eval()
            valid_preds = np.zeros((len(val_loader.dataset), 1))
            avg_val_loss = 0.

            for i, (x_batch, y_batch) in enumerate(val_loader):
                preds = model(x_batch.float().cuda()).detach()
                loss = criterion(preds, y_batch.cuda())

                #preds = torch.sigmoid(preds)
                valid_preds[i * val_loader.batch_size: (i+1) * val_loader.batch_size] = preds.cpu().numpy()

                val_cnt+=1
                writer.add_scalar("val_loss", loss.item(), val_cnt)
                avg_val_loss += loss.item() / len(val_loader)

            err, thresh = compute_eer(val_loader.dataset.y, valid_preds.reshape(-1))
            writer.add_scalar("val_err", err, epoch)

            #if scheduler is not None: scheduler.step()
            torch.save(model.module.state_dict(), self.model_path+str(epoch))
            
            if err < best_err:
                best_epoch = epoch + 1
                torch.save(model.state_dict(), self.model_path)


    def predict_on_test(self, test_loader, model):
        all_outputs, all_fnames = [], []
        model.eval()

        for images, fnames in test_loader:
            preds = model(images.float().cuda()).detach()
            all_outputs.append(preds.cpu().numpy())
            all_fnames.extend(fnames)

        test_preds = pd.DataFrame(data=np.concatenate(all_outputs),
                                  index=all_fnames)
        #test_preds = test_preds.groupby(level=0).mean()
        return test_preds

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def compute_eer(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh