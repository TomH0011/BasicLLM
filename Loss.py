# Want to calculate loss for the logits produced by Unembed
# So text needs to shift one to the right so text[:first_word] = text[:second_word] <- done in Embedding.py
import torch


class CrossEntropyLoss:
    def __init__(self):
        self.targets = None
        self.logits = None
        self.epsilon = 1e-12
        

    # Calculates loss for each target_id not just last vector, then averages it
    # Avoids exponentiating small-large numbers
    # Note: loss = f(W_e, W_Q, W_K, W_V, W_O, W_up, b_up, W_down, b_down, W_u)
    def calulate_cross_entropy_loss_from_logits(self, logits, target_ids):
        # Instead of log(sum(exp(logits))), we compute:
        # max_logit + log(sum(exp(logits - max_logit)))

        max_logits = torch.max(logits, dim=-1, keepdim=True)[0]  # [seq_len, 1]
        shifted_logits = logits - max_logits  # [seq_len, vocab_size]

        # Now exp(shifted_logits) won't overflow since max value is 0
        exp_shifted = torch.exp(shifted_logits)  # [seq_len, vocab_size]
        sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)  # [seq_len, 1]

        # log_sum_exp = max_logit + log(sum_exp)
        log_sum_exp = max_logits + torch.log(sum_exp + self.epsilon)  # [seq_len, 1]
        log_sum_exp = log_sum_exp.squeeze(-1)  # [seq_len]

        target_logits = logits[torch.arange(len(target_ids)), target_ids]  # [seq_len]

        # loss = -log(exp(target_logit) / sum(exp(logits)))
        #      = -(target_logit - log_sum_exp)
        #      = log_sum_exp - target_logit
        loss_per_token = log_sum_exp - target_logits  # [seq_len]

        # Store for backward pass
        self.logits = logits
        self.targets = target_ids

        return loss_per_token.mean()

    # Returns the loss_gradient used for other backward methods
    def backward(self):
        # Delta_Loss / Delta_Logits = 1/N * (P_vec_E_j - target_vec_E_j)
        probs = torch.softmax(self.logits, dim=-1)
        batch_size = self.targets.shape[0]

        probs[torch.arange(batch_size), self.targets] -= 1
        grad_logits = probs / batch_size
        return grad_logits




