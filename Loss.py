# Fixed Loss.py with proper gradient calculation
import torch


class CrossEntropyLoss:
    def __init__(self):
        self.targets = None
        self.logits = None
        self.epsilon = 1e-12

    def calculate_cross_entropy_loss_from_logits(self, logits, target_ids):
        """
        Calculate cross-entropy loss for sequence of logits and targets
        logits: [seq_len, vocab_size]
        target_ids: [seq_len]
        """
        seq_len = logits.shape[0]

        # Numerical stability: subtract max before exp
        max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
        shifted_logits = logits - max_logits

        exp_shifted = torch.exp(shifted_logits)
        sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)

        log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1) + self.epsilon)

        # Get logits for target tokens
        target_logits = logits[torch.arange(seq_len), target_ids]

        # Cross-entropy loss: log_sum_exp - target_logit
        loss_per_token = log_sum_exp - target_logits

        # Store for backward pass
        self.logits = logits
        self.targets = target_ids

        del log_sum_exp, max_logits, exp_shifted, sum_exp, shifted_logits, seq_len

        return loss_per_token.mean()

    def backward(self):
        """
        Compute gradient of loss with respect to logits
        Returns: grad_logits [seq_len, vocab_size]
        """
        seq_len = self.logits.shape[0]

        # Softmax probabilities
        probs = torch.softmax(self.logits, dim=-1)

        # Create one-hot encoding of targets and subtract from probs
        grad_logits = probs.clone()
        grad_logits[torch.arange(seq_len), self.targets] -= 1.0

        # Average over sequence length
        grad_logits = grad_logits / seq_len

        del probs, seq_len

        return grad_logits
