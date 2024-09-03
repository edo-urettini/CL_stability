from avalanche.training.plugins import SupervisedPlugin
import torch
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag, PMatBlockDiag, PMatKFAC, PMatEKFAC, PMatDense, PMatQuasiDiag, PVector
import csv
import copy

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

    def __init__(self, representation, regul, regul_last, alpha_ema, alpha_ema_last, lambda_, clip, variant='classif_logits'):
        super().__init__()
        self.representation = eval(representation)
        self.variant = variant
        self.regul = regul
        self.regul_last = regul_last
        self.alpha_ema = alpha_ema
        self.alpha_ema_last = alpha_ema_last
        self.lambda_ = lambda_
        self.clip = clip
        self.F_ema = None
        self.known_classes = set()
        self.tau = 0
        self.tau_last = 0
        self.gradient_product = None
        self.rho = None
        
        

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

        last_old_layer = list(mat_old.data.keys())[-1]
        last_new_layer = list(mat_new.data.keys())[-1]
        shared_keys = mat_old.data.keys() & mat_new.data.keys()
        
        for layer_id in shared_keys:
            a_old, g_old = mat_old.data[layer_id]
            a_new, g_new = mat_new.data[layer_id]

            ema_a = (1 - self.alpha_ema) * a_old + self.alpha_ema * a_new
            ema_g = (1 - self.alpha_ema) * g_old + self.alpha_ema * g_new

            mat_new.data[layer_id] = (ema_a, ema_g)
        

        if last_old_layer != last_new_layer:
            a_old_last, g_old_last = mat_old.data[last_old_layer]
            a_new_last, g_new_last = mat_new.data[last_new_layer]

            ema_a_last = (1 - self.alpha_ema_last) * a_old_last + self.alpha_ema_last * a_new_last
            ema_g_last = (1 - self.alpha_ema_last) * g_old_last + self.alpha_ema_last * g_new_last[:g_old_last.size(0), :g_old_last.size(1)]

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
        
        
        #Levenberg-Marquardt algorithm with bound
        if self.rho is not None and self.rho > 3/4:
            self.tau = (1 - self.alpha_ema) * self.tau
            self.tau_last = (1 - self.alpha_ema_last) * self.tau_last

        elif self.rho is not None and self.rho < 1/4:
            self.tau = 1/(1 - self.alpha_ema) * self.tau
            self.tau_last = 1/(1 - self.alpha_ema_last) * self.tau_last

        self.tau = max(self.tau, self.regul)
        self.tau = min(self.tau, 10)
        self.tau_last = max(self.tau_last, self.regul_last)
        self.tau_last = min(self.tau_last, 10)


        # Check if new classes are observed. If so, increase the regularization strength by the number of samples from new classes.
        curr_classes = set(strategy.experience.classes_in_this_experience)
        new_classes = curr_classes - self.known_classes    
        if new_classes:
            self.known_classes.update(curr_classes)
       

        #Create a temporary dataloader to compute the FIM
        temp_dataloader = torch.utils.data.TensorDataset(strategy.mb_x, strategy.mb_y)
        temp_dataloader = torch.utils.data.DataLoader(temp_dataloader, batch_size=strategy.mb_x.size(0), shuffle=False)
        
        F = FIM(model=strategy.model,
                loader=temp_dataloader,
                representation=self.representation,
                n_output=strategy.mb_output.size(1),
                variant=self.variant, 
                device=strategy.device,
                lambda_=self.lambda_)



        #Update the EMA of the FIM
        if self.F_ema is None or self.alpha_ema == 1.0:
            self.F_ema = F
        else:
            self.F_ema = self.EMA_kfac(self.F_ema, F)

        original_last_known = torch.norm(strategy.model.linear.classifier.weight.grad[list(self.known_classes), :].flatten())
        original_last_new = torch.norm(strategy.model.linear.classifier.weight.grad[list(new_classes), :].flatten())

        #id of last layer
        last_id = list(F.data.keys())[-1]


        #Compute the regularized gradient
        original_grad_vec = PVector.from_model_grad(strategy.model)
        regularized_grad = self.F_ema.solve(original_grad_vec, regul=self.tau, regul2 = self.tau_last, id = last_id) 
        regularized_grad.to_model_grad(strategy.model)

        #clip gradient norm
        torch.nn.utils.clip_grad_norm_(strategy.model.parameters(), self.clip)
        regularized_grad = PVector.from_model_grad(strategy.model)

        #Assign the dot product between the original gradient and the rescaled regularized gradient (considering also the lr)
        self.gradient_product = original_grad_vec.dot(regularized_grad.__rmul__(strategy.optimizer.param_groups[0]['lr']))
       
        #Register in a csv file the the original gradient norm and the regularized gradient norm, 
        #the original and regularized gradient norm of the last layer (linear.classifier) of the weights connected to known and new classes
        regul_last_known = torch.norm(strategy.model.linear.classifier.weight.grad[list(self.known_classes), :].flatten())
        regul_last_new = torch.norm(strategy.model.linear.classifier.weight.grad[list(new_classes), :].flatten())
        with open('../../tau_grads.csv', mode='a') as file:
            writer = csv.writer(file)
            writer.writerow([self.tau+self.tau_last, original_grad_vec.get_flat_representation().norm().item(), regularized_grad.get_flat_representation().norm().item(), original_last_known.item(), original_last_new.item(), regul_last_known.item(), regul_last_new.item()])


    #Update rho using Levenberg-Marquardt algorithm
    def after_update(self, strategy, **kwargs):
        
        #Compute the actual improvement in the model loss
        loss_after_update = torch.nn.CrossEntropyLoss()(strategy.forward(), strategy.mb_y)
        model_improvement = loss_after_update.item() - strategy.loss.item()
        #Compute the predicted improvement by the quadratic approximation
        improvement_quad = - self.gradient_product / 2

        self.rho = model_improvement / improvement_quad




        

    def compute_vtFv_jacobian(self, model, inputs, v):
        # Create a deep copy of the model to ensure the original model is not modified
        model_copy = copy.deepcopy(model)
        
        # Set the copied model to evaluation mode
        model_copy.eval()

        # Ensure gradients are tracked for model parameters
        for param in model_copy.parameters():
            param.requires_grad = True

        # Check if input vector v is non-zero
        if torch.all(v == 0):
            print("Warning: Input vector v is all zeros.")
            return 0  # Early return since Jv will be zero

        # Flatten model parameters to get a vector representation
        params = torch.nn.utils.parameters_to_vector(model_copy.parameters()).detach()
        
        # Initialize the Jacobian matrix
        num_outputs = model_copy(inputs).numel()  # Total number of outputs
        num_params = params.numel()  # Total number of parameters
        jacobian = torch.zeros(num_outputs, num_params).to(params.device)
        
        # Function that takes model parameters as input and outputs model logits
        def model_outputs_fn(params):
            # Substitute model parameters with the new values in the copied model
            torch.nn.utils.vector_to_parameters(params, model_copy.parameters())
            
            # Forward pass to compute logits
            output = model_copy(inputs)
            
            return output

        # Compute the Jacobian manually
        outputs = model_outputs_fn(params)
        for i in range(num_outputs):
            # Compute gradients of output[i] with respect to all parameters
            grads = torch.autograd.grad(outputs.view(-1)[i], model_copy.parameters(), retain_graph=True)
            # Flatten the gradients and store in the Jacobian matrix
            jacobian[i] = torch.cat([g.view(-1) for g in grads])
        
        # Compute Jv by multiplying the Jacobian with v
        Jv = jacobian @ v  # This computes the Jacobian-vector product

        # Check if Jv is zero after computation
        if torch.all(Jv == 0):
            print("Jv is a zero tensor. Check if gradients are flowing correctly.")
            return 0  # Early return since Jv is zero

        # Compute F_R as the diagonalization of p * (1 - p) where p is the softmax of the logits
        with torch.no_grad():  # We do not need gradients for F_R computation
            outputs = model_copy(inputs)
            p = torch.softmax(outputs, dim=1)
            F_R = torch.diag_embed(p * (1 - p))
        
        # To ensure no gradients are tracked during this process
        with torch.no_grad():
            # Reshape Jv to match the F_R dimensions for batch matrix multiplication
            Jv = Jv.unsqueeze(-1)  # Shape: [batch_size, num_classes, 1]
            
            # Compute F_R * Jv (this is a batch matrix multiplication)
            FR_Jv = torch.bmm(F_R, Jv)  # Shape: [batch_size, num_classes, 1]
            
            # Compute (Jv)^T * (F_R * Jv)
            JvT_FR_Jv = torch.bmm(Jv.transpose(1, 2), FR_Jv)  # Shape: [batch_size, 1, 1]
            
            # Sum across the batch to get the scalar vtFv
            vtFv = JvT_FR_Jv.sum()
        
        return vtFv
        
         



