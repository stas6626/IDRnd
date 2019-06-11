import numpy as np
import torch
import os
import gc
import random
import librosa
import pickle
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# from official code https://colab.research.google.com/drive/1AgPdhSp7ttY18O3fEoHOQKlt_3HJDLi8#scrollTo=cRCaCIb9oguU
def _one_sample_positive_class_precisions(scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
        return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample.
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
            retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
            (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits


def calculate_per_class_lwlrap(truth, scores):
    """Calculate label-weighted label-ranking average precision.

    Arguments:
      truth: np.array of (num_samples, num_classes) giving boolean ground-truth
        of presence of that class in that sample.
      scores: np.array of (num_samples, num_classes) giving the classifier-under-
        test's real-valued score for each class for each sample.

    Returns:
      per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each
        class.
      weight_per_class: np.array of (num_classes,) giving the prior of each
        class within the truth labels.  Then the overall unbalanced lwlrap is
        simply np.sum(per_class_lwlrap * weight_per_class)
    """
    assert truth.shape == scores.shape
    num_samples, num_classes = scores.shape
    # Space to store a distinct precision value for each class on each sample.
    # Only the classes that are true for each sample will be filled in.
    precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
    for sample_num in range(num_samples):
        pos_class_indices, precision_at_hits = (
            _one_sample_positive_class_precisions(scores[sample_num, :],
                                                  truth[sample_num, :]))
        precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
            precision_at_hits)
    labels_per_class = np.sum(truth > 0, axis=0)
    weight_per_class = labels_per_class / float(np.sum(labels_per_class))
    # Form average of each column, i.e. all the precisions assigned to labels in
    # a particular class.
    per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) /
                        np.maximum(1, labels_per_class))
    # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
    #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
    #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
    #                = np.sum(per_class_lwlrap * weight_per_class)
    return per_class_lwlrap, weight_per_class

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    
def load_audio(arr=None, load=False, path=None):
    if load:
        res = []
        for i in arr:
            audio, _ = librosa.core.load(i, sr=44100)
            res.append(audio)
    else:
        with open(path, 'rb') as f:
            res = pickle.load(f)
    return res

class Train:
    def __init__(self, model_path='weight_best.pt',  gradient_acumulation=[]):
        self.model_path = model_path
        self.gradient_acumulation = gradient_acumulation
    
    def fit(self, train_loader, val_loader, model, criterion, optimizer, scheduler, epoches=100):
        writer = SummaryWriter()
        best_epoch = -1
        best_lwlrap = 0.
        acumulate_factor = 1
        tr_cnt, val_cnt = 0, 0  ## TODO
        for epoch in range(epoches):
            model.train()
            avg_loss = 0.
            if epoch in self.gradient_acumulation:
                acumulate_factor *= 2
                
            for x_batch, y_batch in train_loader:
                preds = model(x_batch.float().cuda())
                loss = criterion(preds, y_batch.cuda())
                loss.backward()
                if torch.isnan(loss): print(loss.item())
                if tr_cnt % acumulate_factor == 0:
                    optimizer.step()            
                    if scheduler is not None: scheduler.step();  writer.add_scalar("lr", scheduler.get_lr()[0], tr_cnt) 
                    optimizer.zero_grad()

                tr_cnt += 1
                writer.add_scalar("train_loss", loss.item(), tr_cnt)
            if scheduler is not None: scheduler.step();  writer.add_scalar("lr", scheduler.get_lr()[0], tr_cnt)

            if val_loader is None: continue
            gc.collect()
            model.eval()
            valid_preds = np.zeros((len(val_loader.dataset), 80))
            avg_val_loss = 0.

            for i, (x_batch, y_batch) in enumerate(val_loader):
                preds = model(x_batch.float().cuda()).detach()
                loss = criterion(preds, y_batch.cuda())

                preds = torch.sigmoid(preds)
                valid_preds[i * val_loader.batch_size: (i+1) * val_loader.batch_size] = preds.cpu().numpy()

                val_cnt+=1
                writer.add_scalar("val_loss", loss.item(), val_cnt)
                avg_val_loss += loss.item() / len(val_loader)

            score, weight = calculate_per_class_lwlrap(val_loader.dataset.labels, valid_preds)
            lwlrap = (score * weight).sum()

            writer.add_scalar("val_lwlrap", lwlrap, epoch)
            #if scheduler is not None: scheduler.step()

            if lwlrap > best_lwlrap:
                best_epoch = epoch + 1
                best_lwlrap = lwlrap
                torch.save(model.state_dict(), self.model_path)
        writer.close()      

    def predict_on_test(self, test_loader, model):
        all_outputs, all_fnames = [], []
        model.eval()

        for images, fnames in test_loader:
            preds = torch.sigmoid(model(images.float().cuda()).detach())
            all_outputs.append(preds.cpu().numpy())
            all_fnames.extend(fnames)

        test_preds = pd.DataFrame(data=np.concatenate(all_outputs),
                                  index=all_fnames,
                                  columns=map(str, range(80)))
        test_preds = test_preds.groupby(level=0).mean()

        return test_preds