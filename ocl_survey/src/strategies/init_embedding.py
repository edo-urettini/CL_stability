# Initializing the new weights of the classifier with the embedding of new classes

import torch
from torch import nn
from avalanche.training.plugins import SupervisedPlugin
from torch.nn.functional import avg_pool2d
import torch.nn.functional as F
import os
import csv



class EmbeddingExtractor(nn.Module):
    """
    Wrapper model to extract embeddings using forward hooks. 
    This method works on the specific model modules name. Rename them as needed.
    """

    def __init__(self, model):
        super(EmbeddingExtractor, self).__init__()
        self.model = model
        self.embeddings = None
        self.register_hooks()

    def register_hooks(self):
        def hook(module, input, output):
            self.embeddings = input[0].clone().detach()

        # Register the hook to the last layer of the model
        self.model.linear.register_forward_hook(hook)

    def forward(self, x):
        self.embeddings = None
        out = self.model(x)
        return self.embeddings

    


class InitEmbeddingPlugin(SupervisedPlugin):
    """
    SupervisedPlugin for initializing the new weights of the classifier with the embedding of new classes.
    """

    def __init__(self):
        super().__init__()
        self.known_classes = set()
        self.class_counts = {}
        self.class_embeddings = {}
        self.required_nsamples = 10


        
    @torch.no_grad()
    def before_training_iteration(self, strategy, **kwargs):
        if not strategy.train:
            return
        
        classifier = strategy.model.linear.classifier  
        device = strategy.device


        # Find newly added classes (assuming masking is active in model.adaptation)
        curr_classes = set(strategy.experience.classes_in_this_experience)
        new_classes = curr_classes - self.known_classes

        if not new_classes:
            return
        
        #Update known classes
        #self.known_classes.update(curr_classes)

        # Forward the current batch to get all embeddings
        emb_model = EmbeddingExtractor(strategy.model)
        emb_model.to(device)
        emb_model.train()
        embeddings = emb_model(strategy.mb_x.to(device))

        #assert if the embeddings are correct  
        original_output = strategy.model(strategy.mb_x.to(device))
        output_from_emb = strategy.model.linear(embeddings)
        assert torch.allclose(original_output, output_from_emb, atol=1e-6), "Embeddings are not correct"


        labels = strategy.mb_y.to(device)

        # Initialize the new classifier weights with the mean of the last n embeddings of each new class
        for class_idx in new_classes:
            class_mask = labels == class_idx
            class_embeddings = embeddings[class_mask]
            if class_embeddings.size(0) > 0:
                class_embeddings = F.normalize(class_embeddings, p=2, dim=1)
                if class_idx not in self.class_counts:
                    self.class_counts[class_idx] = 0
                    self.class_embeddings[class_idx] = torch.zeros(class_embeddings.size(1)).to(device)
                for emb in class_embeddings:
                    if self.class_counts[class_idx] < self.required_nsamples:
                        self.class_embeddings[class_idx] = (self.class_embeddings[class_idx] * self.class_counts[class_idx] + emb) / (self.class_counts[class_idx] + 1)
                        self.class_counts[class_idx] += 1
                        classifier.weight[class_idx] = self.class_embeddings[class_idx]
                        classifier.bias[class_idx] = 0
                    if self.class_counts[class_idx] == self.required_nsamples:
                        self.known_classes.add(class_idx)
                        classifier.weight[class_idx] = self.class_embeddings[class_idx]
                        classifier.bias[class_idx] = 0
  
    #Stop the gradient flow for new classes
    def before_backward(self, strategy, **kwargs):
        outputs = strategy.mb_output
        def backward_hook(grad):
            for class_idx in range(grad.size(1)):
                if self.class_counts.get(class_idx, 0) < self.required_nsamples:
                    grad[:, class_idx] = 0
            return grad
        outputs.register_hook(backward_hook)

