# Want to calculate loss for the logits produced by Unembed
# So text needs to shift one to the right so text[:first_word] = text[:second_word] <- done in Embedding.py
import torch


class CrossEntropyLoss:
    def __init__(self):
        self.epsilon = 1e-12

    # Calculates loss for each target_id not just last vector, then averages it
    # Avoids exponentiating small-large numbers
    def calulate_cross_entropy_loss_from_logits(self, logits, target_ids):
        log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))      # [seq_len]
        target_logits = logits[torch.arange(len(target_ids)), target_ids]  # [seq_len]
        loss = - (target_logits - log_sum_exp)                             # [seq_len]
        return loss.mean()

