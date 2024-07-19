from avalanche.training.plugins import SupervisedPlugin
import torch
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag, PMatBlockDiag, PMatKFAC, PMatEKFAC, PMatDense, PMatQuasiDiag, PVector


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

    def __init__(self, representation, regul, alpha_ema, variant='classif_logits'):
        super().__init__()
        self.representation = eval(representation)
        self.variant = variant
        self.regul = regul
        self.alpha_ema = alpha_ema
        self.F_ema = None
        self.known_classes = set()
        self.tau = 0.0
        self.gradient_product = None
        self.rho = None


    def EMA_kfac(self, mat_old, mat_new, alpha):
        """
        Compute the exponential moving average of two PMatKFAC matrices.

        :param mat_old: The previous PMatKFAC matrix.
        :param mat_new: The new PMatKFAC matrix.
        :param alpha: The smoothing factor for the EMA. Weight for new data.
        :return: A new PMatKFAC matrix representing the EMA.
        """

        last_old_layer = list(mat_old.data.keys())[-1]
        last_new_layer = list(mat_new.data.keys())[-1]
        shared_keys = mat_old.data.keys() & mat_new.data.keys()
        
        for layer_id in shared_keys:
            a_old, g_old = mat_old.data[layer_id]
            a_new, g_new = mat_new.data[layer_id]

            ema_a = (1 - alpha) * a_old + alpha * a_new
            ema_g = (1 - alpha) * g_old + alpha * g_new

            mat_new.data[layer_id] = (ema_a, ema_g)
        
        if last_old_layer != last_new_layer:
            a_old_last, g_old_last = mat_old.data[last_old_layer]
            a_new_last, g_new_last = mat_new.data[last_new_layer]

            ema_a_last = (1 - alpha) * a_old_last + alpha * a_new_last
            ema_g_last = (1 - alpha) * g_old_last + alpha * g_new_last[:g_old_last.size(0), :g_old_last.size(1)]

            g_new_last[:g_old_last.size(0), :g_old_last.size(1)] = ema_g_last

            mat_new.data[last_new_layer] = (ema_a_last, g_new_last)

        # Create a new PMatKFAC instance with the EMA data
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
        #Levenberg-Marquardt algorithm
        if self.rho is not None and self.rho > 3/4:
            self.tau = (1 - self.alpha_ema) * self.tau

        elif self.rho is not None and self.rho < 1/4:
            self.tau = 1/(1 - self.alpha_ema) * self.tau
        '''''

        # Check if new classes are observed. If so, increase the regularization strength by the number of samples from new classes.
        curr_classes = set(strategy.experience.classes_in_this_experience)
        new_classes = curr_classes - self.known_classes    
        if new_classes:
            #Count the number of samples in mb_y that correspond to new classes
            new_samples = 0
            for y in strategy.mb_y:
                if y.item() in new_classes:
                    new_samples += 1
            
            self.tau += 1 * new_samples
            self.known_classes.update(curr_classes)
        else:
            self.tau *= (1-self.alpha_ema)
        

        #Create a temporary dataloader to compute the FIM
        temp_dataloader = torch.utils.data.TensorDataset(strategy.mb_x, strategy.mb_y)
        temp_dataloader = torch.utils.data.DataLoader(temp_dataloader, batch_size=strategy.mb_x.size(0), shuffle=False)
        
        F = FIM(model=strategy.model,
                loader=temp_dataloader,
                representation=self.representation,
                n_output=strategy.mb_output.size(1),
                variant=self.variant, 
                device=strategy.device)
        #Update the EMA of the FIM
        if self.F_ema is None or self.alpha_ema == 1.0:
            self.F_ema = F
        else:
            self.F_ema = self.EMA_kfac(self.F_ema, F, self.alpha_ema)
        

        original_grad_vec = PVector.from_model_grad(strategy.model)
        regularized_grad = self.F_ema.solve(original_grad_vec, regul=self.regul + self.tau) 
        regularized_grad.to_model_grad(strategy.model)  
'''''
        #Compute the dot product between the original gradient and the regularized gradient
        self.gradient_product = torch.dot(original_grad_vec.get_flat_representation(), regularized_grad.get_flat_representation())

    #Compute rho as the ratio between the improvement in the loss of the model itself and of its quadratic approximation
    def after_update(self, strategy, **kwargs):

        loss_after_update = torch.nn.CrossEntropyLoss()(strategy.forward(), strategy.mb_y)
        model_improvement = loss_after_update.item() - strategy.loss.item()
        lr = strategy.optimizer.param_groups[0]['lr']
        improvement_quad = (lr**2 / 2 - lr) * self.gradient_product
        self.rho = model_improvement / improvement_quad

'''''