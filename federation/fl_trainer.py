import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch
import torch.nn as nn
import torch.optim as optim

from data.partition        import partition_into_hospitals, get_reference_dataset
from data.dataset          import get_dataloader
from models.teacher        import TeacherModel
from models.student        import StudentModel
from distillation.kd_loss  import KnowledgeDistillationLoss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_class_weights(hospital_dict):
    """
    Computes inverse-frequency class weights to handle imbalanced data.
    Minority class gets higher weight → model penalized more for missing it.
    """
    labels = hospital_dict['labels']
    n_total     = len(labels)
    n_normal    = labels.count(0)
    n_pneumonia = labels.count(1)

    # Avoid division by zero
    w_normal    = n_total / (2 * n_normal)    if n_normal    > 0 else 1.0
    w_pneumonia = n_total / (2 * n_pneumonia) if n_pneumonia > 0 else 1.0

    return torch.tensor([w_normal, w_pneumonia], dtype=torch.float32)

def train_teacher_local(model, dataloader, epochs=1, lr=1e-4):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model, total_loss / len(dataloader)


def train_student_local(student, expert_models, dataloader,
                        epochs=1, lr=3e-4, temperature=4.0, alpha=0.4,
                        class_weights=None):
    student.train()
    for m in expert_models:
        m.eval()

    optimizer  = optim.Adam(student.parameters(), lr=lr)

    # Weighted cross entropy for imbalanced hospitals
    if class_weights is not None:
        ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    else:
        ce_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    for _ in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            student_logits = student(images)

            with torch.no_grad():
                teacher_logits_list = [m(images) for m in expert_models]
                teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)

            # Hard loss — weighted CE for imbalanced classes
            hard_loss = ce_loss_fn(student_logits, labels)

            # Distillation loss — KL divergence with temperature scaling
            soft_teacher = torch.nn.functional.softmax(
                teacher_logits / temperature, dim=1)
            soft_student = torch.nn.functional.log_softmax(
                student_logits / temperature, dim=1)
            distill_loss = torch.nn.functional.kl_div(
                soft_student, soft_teacher, reduction='batchmean'
            ) * (temperature ** 2)

            # Combined loss
            loss = (1 - alpha) * hard_loss + alpha * distill_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return student, total_loss / len(dataloader)

def evaluate_model(model, dataloader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds    = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else 0.0


def fedavg(global_model, local_models, sample_counts):
    total = sum(sample_counts)
    agg   = {}
    for k, v in global_model.state_dict().items():
        if v.dtype in (torch.long, torch.int32, torch.int64, torch.bool):
            # Integer buffers (e.g. num_batches_tracked) — copy from first model
            agg[k] = local_models[0].state_dict()[k].clone()
        else:
            # Float parameters — weighted average
            agg[k] = torch.zeros_like(v, dtype=torch.float32)
            for model, n in zip(local_models, sample_counts):
                agg[k] += (n / total) * model.state_dict()[k].float()
    global_model.load_state_dict(agg)
    return global_model


def run_federated_learning(data_dir, num_rounds=5, local_epochs=1,
                           temperature=4.0, alpha=0.4):

    print(f"\n{'='*60}")
    print(f"  Federated Learning with Knowledge Distillation")
    print(f"  Device: {DEVICE}")
    print(f"  Rounds: {num_rounds} | Local Epochs: {local_epochs}")
    print(f"  KD Temperature: {temperature} | Alpha: {alpha}")
    print(f"{'='*60}\n")

    hospitals     = partition_into_hospitals(data_dir)
    train_loaders = [get_dataloader(h, batch_size=32, augment=True)
                     for h in hospitals]

    global_teacher = TeacherModel(num_classes=2, pretrained=True).to(DEVICE)
    global_student = StudentModel(num_classes=2).to(DEVICE)

    teacher_indices = [i for i, h in enumerate(hospitals) if     h['is_expert']]
    student_indices = [i for i, h in enumerate(hospitals) if not h['is_expert']]

    history = {
        'round'             : [],
        'expert_avg_acc'    : [],
        'nonexpert_avg_acc' : [],
        'expert_avg_loss'   : [],
        'nonexpert_avg_loss': [],
        'per_hospital_acc'  : [],   # track each hospital individually
    }

    for round_num in range(1, num_rounds + 1):
        print(f"── Round {round_num}/{num_rounds} " + "─"*40)

        # Step 1: Train expert hospitals
        print("  [1/3] Training expert hospitals...")
        local_teacher_models = []
        teacher_losses       = []
        for i in teacher_indices:
            m, loss = train_teacher_local(
                copy.deepcopy(global_teacher),
                train_loaders[i], epochs=local_epochs
            )
            local_teacher_models.append(m)
            teacher_losses.append(loss)
            print(f"        Hospital {i} (Expert)     | Loss: {loss:.4f}")

        # Step 2: Train non-expert hospitals with KD
        print("  [2/3] Training non-expert hospitals with KD...")
        local_student_models = []
        student_losses       = []
        # Main training loop — Step 2
        for i in student_indices:
            weights = get_class_weights(hospitals[i]).to(DEVICE)
            m, loss = train_student_local(
                copy.deepcopy(global_student),
                local_teacher_models,
                train_loaders[i],
                epochs=local_epochs,
                temperature=temperature,
                alpha=alpha,
                class_weights=weights
            )
            local_student_models.append(m)
            student_losses.append(loss)
            print(f"        Hospital {i} (Non-Expert) | Loss: {loss:.4f}")

        # Step 3: FedAvg aggregation
        print("  [3/3] Aggregating models (FedAvg)...")
        global_teacher = fedavg(global_teacher, local_teacher_models,
                                [hospitals[i]['total'] for i in teacher_indices])
        global_student = fedavg(global_student, local_student_models,
                                [hospitals[i]['total'] for i in student_indices])

        # ── Personalized evaluation ──────────────────────────────────────
        # Each hospital fine-tunes the global model briefly on local data
        # then we evaluate that personalized version
        print("     Personalizing and evaluating per hospital...")

        expert_accs   = []
        nonexpert_accs = []
        per_hospital  = {}

        for idx, i in enumerate(teacher_indices):
            # Fine-tune global teacher on local data (1 epoch)
            personalized, _ = train_teacher_local(
                copy.deepcopy(global_teacher),
                train_loaders[i], epochs=1, lr=5e-5
            )
            acc = evaluate_model(personalized, train_loaders[i])
            expert_accs.append(acc)
            per_hospital[f'H{i}_expert'] = acc

        for idx, i in enumerate(student_indices):
            weights = get_class_weights(hospitals[i]).to(DEVICE)
            personalized, _ = train_student_local(
                copy.deepcopy(global_student),
                local_teacher_models,
                train_loaders[i],
                epochs=3,           # increased from 1 to 3
                temperature=temperature,
                alpha=alpha,
                class_weights=weights
            )
            acc = evaluate_model(personalized, train_loaders[i])
            nonexpert_accs.append(acc)
            per_hospital[f'H{i}_nonexpert'] = acc

        avg_exp_acc   = sum(expert_accs)    / len(expert_accs)
        avg_nexp_acc  = sum(nonexpert_accs) / len(nonexpert_accs)
        avg_exp_loss  = sum(teacher_losses) / len(teacher_losses)
        avg_nexp_loss = sum(student_losses) / len(student_losses)

        print(f"\n  📊 Round {round_num} Summary:")
        print(f"     Expert Avg Accuracy    : {avg_exp_acc:.4f}")
        print(f"     Non-Expert Avg Accuracy: {avg_nexp_acc:.4f}")
        print(f"     Expert Avg Loss        : {avg_exp_loss:.4f}")
        print(f"     Non-Expert Avg Loss    : {avg_nexp_loss:.4f}")
        print(f"     Per-Hospital Accuracy  : {per_hospital}\n")

        history['round'].append(round_num)
        history['expert_avg_acc'].append(avg_exp_acc)
        history['nonexpert_avg_acc'].append(avg_nexp_acc)
        history['expert_avg_loss'].append(avg_exp_loss)
        history['nonexpert_avg_loss'].append(avg_nexp_loss)
        history['per_hospital_acc'].append(per_hospital)

    print(f"{'='*60}")
    print("  Training Complete")
    print(f"{'='*60}\n")

    return global_teacher, global_student, history


if __name__ == "__main__":
    DATA_DIR = r"G:\PROJECTS\FL with KD\Kaggle (Pneumonia) Chest X-ray Dataset\chest_xray\train"

    teacher, student, history = run_federated_learning(
        data_dir     = DATA_DIR,
        num_rounds   = 3,
        local_epochs = 1,
        temperature  = 4.0,
        alpha        = 0.4
    )

    print("Final History:")
    for i, r in enumerate(history['round']):
        print(f"  Round {r} | Expert Acc: {history['expert_avg_acc'][i]:.4f} | "
              f"Non-Expert Acc: {history['nonexpert_avg_acc'][i]:.4f}")