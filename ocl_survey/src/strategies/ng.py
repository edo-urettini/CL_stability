from avalanche.training.plugins import SupervisedPlugin
import torch
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag, PMatBlockDiag, PMatKFAC, PMatEKFAC, PMatDense, PMatQuasiDiag, PVector

import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import os
import csv
import time




class NGPlugin(SupervisedPlugin):
    """
    SupervisedPlugin for Fisher Information Regularization.

    This plugin modifies gradients before optimizer update by
    multiplying them with the inverse of the Fisher Information matrix (FIM).

    Args:
        model (torch.nn.Module): The model to be trained.
        representation (str, optional): The model representation to use for 
            FIM calculation (e.g., 'PMatKFAC'). Defaults to PMatKFAC.
        n_output (int, optional): The number of output classes. Defaults to None.
        variant (str, optional): The variant of the FIM calculation (e.g., 'classif_logits').
            Defaults to 'classif_logits'.
    """

    def __init__(self, representation, regul, variant='classif_logits'):
        super().__init__()
        self.representation = eval(representation)
        self.variant = variant
        self.regul = regul

    def before_update(self, strategy, **kwargs):
        """
        Modifies gradients before optimizer update.

        Args:
            strategy (Template): The current training strategy.
        """
        if not strategy.train:
            return

        #If batch size is equal to 20, compute the gradients for the first 10 samples and the last 10 samples
        if strategy.mb_x.size(0) == 20:
            
            # Compute gradients for the first 10 samples
            temp_model = copy.deepcopy(strategy.model)
            temp_res = temp_model(strategy.mb_x[:10])
            temp_criterion = torch.nn.CrossEntropyLoss()
            temp_loss = temp_criterion(temp_res, strategy.mb_y[:10])
            temp_loss.backward()
            grad_first = parameters_to_vector([param.grad for param in temp_model.parameters() if param.grad is not None])
            grad_norm_first = torch.norm(grad_first, p=2).item()
            
            # Reset gradients for the temporary model
            temp_model.zero_grad()
            
            # Compute gradients for the last 10 samples
            temp_res = temp_model(strategy.mb_x[10:])
            temp_loss = temp_criterion(temp_res, strategy.mb_y[10:])
            temp_loss.backward()
            grad_last = parameters_to_vector([param.grad for param in temp_model.parameters() if param.grad is not None])
            grad_norm_last = torch.norm(grad_last, p=2).item()

            #compute cosine similarity between the two gradients
            cos = torch.nn.functional.cosine_similarity(grad_first, grad_last, dim=0).item()

            # Log the norms to the CSV file
            csv_file = '../../grad_data.csv'
            file_exists = os.path.isfile(csv_file)
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['New_data_Grad_Norm', 'Buffer_data_Grad_Norm', 'Cosine_Similarity'])
                writer.writerow([grad_norm_first, grad_norm_last, cos])
        
        elif strategy.mb_x.size(0) == 40:
            strategy.model.zero_grad()

            res = strategy.model(strategy.mb_x[20:])
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(res, strategy.mb_y[20:])
            loss.backward()

            #Create a temporary dataloader to calculate the FIM
            temp_dataloader = torch.utils.data.TensorDataset(strategy.mb_x[20:], strategy.mb_y[20:])
            temp_dataloader = torch.utils.data.DataLoader(temp_dataloader, batch_size=int(strategy.mb_x.size(0)/2), shuffle=False)


            F = FIM(model=strategy.model,
                    loader=temp_dataloader,
                    representation=self.representation,
                    n_output=strategy.mb_output.size(1),
                    variant=self.variant, 
                    device=strategy.device)

            original_grad_vec = PVector.from_model_grad(strategy.model)
            regularized_grad = F.solve(original_grad_vec, regul=self.regul) 
            regularized_grad.to_model_grad(strategy.model)  

