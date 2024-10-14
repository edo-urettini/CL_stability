from avalanche.training.plugins import SupervisedPlugin
import torch
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag, PMatBlockDiag, PMatKFAC, PMatEKFAC, PMatDense, PMatQuasiDiag, PVector
import csv
import copy

class SignSGDPlugin(SupervisedPlugin):
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

    def __init__(self, representation, regul, regul_last, alpha_ema, alpha_ema_last, lambda_, clip, num_task_per_exp, buffer_idx, variant='classif_logits'):
        super().__init__()
        self.representation = eval(representation)
        self.variant = variant
        self.regul = regul
        self.regul_last = regul_last
        self.alpha_ema = alpha_ema
        #Warning: alpha_ema_last not used in the current implementation
        self.alpha_ema_last = alpha_ema
        self.lambda_ = lambda_
        self.clip = clip
        self.F_ema = None
        self.F_ema_inv = None
        self.known_classes = set()
        self.tau = 0
        self.gradient_product = None
        self.rho = None
        self.n_new = num_task_per_exp
        self.output_size = None
        self.iterations = 0
        self.buffer_idx = buffer_idx
        
        

        # If file csv file exists, delete it
        import os
        file_path = '../../tau_grads.csv'
        if os.path.exists(file_path):
            os.remove(file_path)
        # Create a new csv file with the header
        with open('../../tau_grads.csv', mode='w') as file:
            writer = csv.writer(file)
            writer.writerow(['tau', 'original_grad_norm', 'regularized_grad_norm', 'original_last_known', 'original_last_new', 'regul_last_known', 'regul_last_new'])




    def EMA_kfac(self, mat_old, mat_new):
        """
        Compute the exponential moving average of two PMatKFAC matrices.

        :param mat_old: The previous PMatKFAC matrix.
        :param mat_new: The new PMatKFAC matrix.
        :return: A new PMatKFAC matrix representing the EMA.
        """
        if self.representation == PMatEKFAC:
            old = mat_old.data[0]
            new = mat_new.data[0]
        else:
            old = mat_old.data
            new = mat_new.data

        last_old_layer = list(old.keys())[-1]
        last_new_layer = list(new.keys())[-1]
        shared_keys = old.keys() & new.keys()
        
        for layer_id in shared_keys:
            a_old, g_old = old[layer_id]
            a_new, g_new = new[layer_id]

            ema_a = (1 - self.alpha_ema) * a_old + self.alpha_ema * a_new
            ema_g = (1 - self.alpha_ema) * g_old + self.alpha_ema * g_new

            new[layer_id] = (ema_a, ema_g)
        
       
        if last_old_layer != last_new_layer and self.alpha_ema_last < 1.0:
            a_old_last, g_old_last = old[last_old_layer]
            a_new_last, g_new_last = new[last_new_layer]

            ema_a_last = (1 - self.alpha_ema_last) * a_old_last + self.alpha_ema_last * a_new_last
            
            #g_new_last = g_new_last * self.alpha_ema_last + (1 - self.alpha_ema_last) 

            new[last_new_layer] = (ema_a_last, g_new_last)

        if self.representation == PMatEKFAC:
            mat_new.data = (new, mat_new.data[1])
        else:
            mat_new.data = new
 
         # Create a new PMatKFAC instance with the EMA data
        return mat_new
    
    def EMA_diag(self, diag_old, mat_new):
        #Compute the EMA of the diagonal of the FIM when using PMatEkfac representation
        old = diag_old
        new = mat_new.data[1]

        shared_keys = old.keys() & new.keys()
        last_old_layer = list(old.keys())[-1]
        last_new_layer = list(new.keys())[-1]

        for layer_id in shared_keys:
            old_diag = old[layer_id]
            new_diag = new[layer_id]

            ema_diag = (1 - self.alpha_ema) * old_diag + self.alpha_ema * new_diag
            new[layer_id] = ema_diag

        mat_new.data = (mat_new.data[0], new)

        return mat_new
  

    def before_update(self, strategy, **kwargs):
        """
        Modifies gradients before optimizer update.

        Args:
            strategy (Template): The current training strategy.
        """
        if not strategy.train:
            return 
        
        '''''
        #Levenberg-Marquardt algorithm with bound
        if self.rho is not None and self.rho > 3/4:
            self.tau = (1 - self.alpha_ema) * self.tau
            self.tau_last = (1 - self.alpha_ema_last) * self.tau_last

        elif self.rho is not None and self.rho < 1/4:
            self.tau = 1/(1 - self.alpha_ema) * self.tau
            self.tau_last = 1/(1 - self.alpha_ema_last) * self.tau_last
        '''''

        # Check if new classes are observed
        curr_classes = set(strategy.experience.classes_in_this_experience)
        new_classes = curr_classes - self.known_classes    
        if new_classes:
            self.known_classes.update(curr_classes)


        #Compute the weights for the FIM to compensate for different classes frequencies
        batch_y = torch.cat((strategy.mb_y, strategy.mb_buffer_y))
        batch_x = torch.cat((strategy.mb_x, strategy.mb_buffer_x))
        batch_size = int(batch_x.size(0))
        weights = torch.ones(batch_size, device=strategy.device)
        n_known = len(self.known_classes)
        if len(self.buffer_idx) == len(weights):
            weights[self.buffer_idx] = n_known / self.n_new 



        if self.tau == 0:
            self.tau = strategy.optimizer.param_groups[0]['lr']
        else:
            self.tau = self.tau + self.regul


        #Create a temporary dataloader to compute the FIM
        temp_dataloader = torch.utils.data.TensorDataset(batch_x, batch_y)
        temp_dataloader = torch.utils.data.DataLoader(temp_dataloader, batch_size=batch_size, shuffle=False)

        if self.representation == PMatEKFAC and self.F_ema is not None:
            old_diag = self.F_ema.data[1]
        else:
            old_diag = None

        if strategy.mb_output.size(1) != self.output_size:
            self.iterations = 0
      
        if self.iterations % strategy.train_epochs == 0:
            #Compute and update the FIM
            F = FIM(model=strategy.model,
                    loader=temp_dataloader,
                    representation=self.representation,
                    n_output=strategy.mb_output.size(1),
                    variant=self.variant, 
                    device=strategy.device,
                    lambda_=self.lambda_, 
                    weights=weights)

            #Update the EMA of the FIM
            if self.F_ema is None or (self.alpha_ema == 1.0 and self.alpha_ema_last == 1.0):
                self.F_ema = F
            else:
                self.F_ema = self.EMA_kfac(self.F_ema, F)

            self.F_ema_inv = self.F_ema.inverse(regul = self.tau)

        self.iterations += 1

        if self.representation == PMatEKFAC:
            self.F_ema.update_diag(temp_dataloader)
            if old_diag is not None:
                self.F_ema = self.EMA_diag(old_diag, self.F_ema)

        #original_last_known = torch.norm(strategy.model.linear.classifier.weight.grad[list(self.known_classes), :].flatten())
        #original_last_new = torch.norm(strategy.model.linear.classifier.weight.grad[list(new_classes), :].flatten())

        #Size of the output layer
        self.output_size = strategy.mb_output.size(1)


        #Compute the regularized gradient
        original_grad_vec = PVector.from_model_grad(strategy.model)
        regularized_grad = self.F_ema_inv.mv(original_grad_vec)
        regularized_grad.to_model_grad(strategy.model)

