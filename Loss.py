# Want to calculate loss for the logits produced by Unembed
# So text needs to shift one to the right so text[:first_word] = text[:second_word] <- done in Embedding.py
import torch


class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = 1e-12

    # Calculates loss for each target_id not just last vector, then averages it
    # Avoids exponentiating small-large numbers
    # Note: loss = f(W_e, W_Q, W_K, W_V, W_O, W_up, b_up, W_down, b_down, W_u)
    def calulate_cross_entropy_loss_from_logits(self, logits, target_ids):
        log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))      # [seq_len]
        target_logits = logits[torch.arange(len(target_ids)), target_ids]  # [seq_len]
        loss = - (target_logits - log_sum_exp)                             # [seq_len]
        self.logits = logits
        self.targets = target_ids
        return loss.mean()

    # Returns the loss_gradient used for other backward methods
    def backward(self):
        # Delta_Loss / Delta_Logits = 1/N * (P_vec_E_j - target_vec_E_j)
        probs = torch.softmax(self.logits, dim=-1)
        batch_size = self.targets.shape[0]
        probs[torch.arange(batch_size), self.targets] -= 1
        grad_logits = probs / batch_size
        return grad_logits




