import os
import wandb
import torch
import numpy as np
import torch.nn as nn
import logging
class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_g_paths = []
        self.top_model_d_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model_g, model_d, epoch, metric_val):
        model_g_path = os.path.join(self.dirpath, model_g.__class__.__name__ + f'_epoch{epoch}.pt')
        model_d_path = os.path.join(self.dirpath, model_d.__class__.__name__ + f'_epoch{epoch}.pt')

        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save: 
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_g_path} and {model_d_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model_g.state_dict(), model_g_path)
            torch.save(model_d.state_dict(),model_d_path)
            self.log_artifact(f'model-g-ckpt-epoch-{epoch}.pt', model_g_path, metric_val)
            self.log_artifact(f'model-d-ckpt-epoch-{epoch}.pt', model_d_path, metric_val)

            self.top_model_g_paths.append({'path': model_g_path, 'score': metric_val})
            self.top_model_d_paths.append({'path': model_g_path, 'score': metric_val})

            self.top_model_g_paths = sorted(self.top_model_g_paths, key=lambda o: o['score'], reverse=not self.decreasing)
            self.top_model_d_paths = sorted(self.top_model_d_paths, key=lambda o: o['score'], reverse=not self.decreasing)

        if len(self.top_model_g_paths)>self.top_n: 
            self.cleanup()
        if len(self.top_model_d_paths)>self.top_n:
            self.cleanup()
    
    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)        
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]