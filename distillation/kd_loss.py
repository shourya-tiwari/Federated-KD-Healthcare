import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Combined loss for student model training.

    Total Loss = (1 - alpha) * Hard Loss
               +      alpha  * T^2 * Distillation Loss

    Args:
        temperature : T — controls softness of predictions (default 4)
        alpha       : weight for distillation loss (default 0.7)

    Higher temperature → softer probability distributions → 
    more knowledge transferred from teacher to student.
    """

    def __init__(self, temperature=4.0, alpha=0.7):
        super(KnowledgeDistillationLoss, self).__init__()
        self.T     = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, true_labels):
        """
        Args:
            student_logits : raw output from student model  (batch, num_classes)
            teacher_logits : raw output from teacher model  (batch, num_classes)
            true_labels    : ground truth labels            (batch,)

        Returns:
            total_loss        : combined loss scalar
            hard_loss_val     : cross entropy component (for logging)
            distill_loss_val  : KL divergence component (for logging)
        """

        # --- Hard Loss ---
        # Normal cross entropy between student predictions and true labels
        hard_loss = self.ce_loss(student_logits, true_labels)

        # --- Distillation Loss ---
        # Soften both teacher and student outputs with temperature
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)

        # KL Divergence: how different is student from teacher
        # kl_div expects (log_predictions, targets)
        distill_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction='batchmean'
        ) * (self.T ** 2)  # T^2 scaling to normalize gradients

        # --- Combined Loss ---
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * distill_loss

        return total_loss, hard_loss.item(), distill_loss.item()


if __name__ == "__main__":
    import torch

    print("Testing Knowledge Distillation Loss")
    print("=" * 45)

    batch_size  = 8
    num_classes = 2

    # Simulate teacher logits (confident predictions)
    teacher_logits = torch.tensor([
        [ 2.5, -1.0],  # strongly Normal
        [-1.0,  2.8],  # strongly Pneumonia
        [ 1.8, -0.5],  # Normal
        [-0.3,  1.9],  # Pneumonia
        [ 2.1, -0.8],  # Normal
        [-1.2,  2.5],  # Pneumonia
        [ 0.9,  0.1],  # weakly Normal
        [-0.5,  1.2],  # weakly Pneumonia
    ])

    # Simulate student logits (less confident — it's still learning)
    student_logits = torch.tensor([
        [ 1.2, -0.3],
        [-0.4,  1.1],
        [ 0.8,  0.2],
        [ 0.1,  0.9],
        [ 1.5, -0.2],
        [-0.6,  1.3],
        [ 0.3,  0.4],
        [-0.1,  0.7],
    ])

    # True labels
    true_labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    # Test with different temperature and alpha values
    configs = [
        (4.0, 0.7),  # default
        (2.0, 0.5),  # lower temperature, balanced
        (6.0, 0.9),  # high temperature, teacher-heavy
    ]

    for T, alpha in configs:
        kd_loss = KnowledgeDistillationLoss(temperature=T, alpha=alpha)
        total, hard, distill = kd_loss(student_logits, teacher_logits, true_labels)

        print(f"T={T}, alpha={alpha}")
        print(f"  Hard Loss:        {hard:.4f}")
        print(f"  Distillation Loss:{distill:.4f}")
        print(f"  Total Loss:       {total:.4f}")
        print()

    print("=" * 45)
    print("KD Loss working correctly.")