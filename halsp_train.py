# train.py
# Training script for HALSP-Net (ResNet50 variant) on CIFAR-100.
# Implements the training recipe described in the HALSP paper:
#   - 10 epoch linear warmup, followed by 190 epoch cosine annealing.
#   - Label smoothing 0.1, TrivialAugment, SGD with Nesterov momentum.
#   - Three training phases: warmup (dense), search (sparse active channels), cooldown (dense).
#   - Periodic topology maintenance using optimizer momentum.

import os
import time
import random
import logging
import subprocess
import resource

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ----------------------------------------------------------------------
# Import HALSP model
# ----------------------------------------------------------------------
try:
    from halsp import ResNet50
except ImportError:
    raise ImportError("ERROR: 'halsp.py' not found!")

# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------
LEARNING_RATE = 0.2
BATCH_SIZE = 256
EPOCHS = 200
NUM_CLASSES = 100
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
ACCUMULATION_STEPS = 1

SAVE_DIR = "/results"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
log_filename = "training_log.txt"
if os.path.exists(log_filename):
    os.remove(log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("================================================================")
logger.info(f"STARTING TRAIN (LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}, Accum: {ACCUMULATION_STEPS})")
logger.info(f"Device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("Can not find GPU, working in CPU mode.")
logger.info("================================================================\n")

# ----------------------------------------------------------------------
# Data preparation (CIFAR-100)
# ----------------------------------------------------------------------
logger.info("Preparing Dataset...")

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)

from torchvision.transforms import autoaugment

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),   # TrivialAugment (TA)
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std),
])

trainset = torchvision.datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=transform_train
)
testset = torchvision.datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=transform_test
)

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

# ----------------------------------------------------------------------
# Model, initialization, and loss
# ----------------------------------------------------------------------
model = ResNet50(num_classes=NUM_CLASSES).to(device)

# Additional initializations (overwrites some from model init, kept as in original code)
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.GroupNorm):
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.normal_(m.bias, mean=0.0, std=0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

nn.init.normal_(model.fc.weight, 0, 0.01)
nn.init.constant_(model.fc.bias, 0)

# Label smoothing cross-entropy (ε=0.1)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ----------------------------------------------------------------------
# Optimizer & parameter groups (no weight decay on bias/norm params)
# ----------------------------------------------------------------------
def get_parameter_groups(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay}
    ]

params = get_parameter_groups(model, weight_decay=WEIGHT_DECAY)

optimizer = optim.SGD(
    params,
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    nesterov=True
)

# LR schedule: linear warmup (10 epochs) + cosine annealing (190 epochs)
scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=10
)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=190,
    eta_min=0.0001
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_cosine],
    milestones=[10]
)

# ----------------------------------------------------------------------
# Metric helper
# ----------------------------------------------------------------------
def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
base_header = f"{'Epoch':<6} | {'Batch':<6} | {'Loss':<8} | {'Acc@1':<8} | {'LR':<9} | {'GradNorm':<10}"
logger.info(base_header)
logger.info("-" * len(base_header))

total_training_start = time.time()
best_acc = 0.0
current_grad_norm = 0.0

for epoch in range(EPOCHS):
    # ----- Phase management -----
    if epoch == 0:
        logger.info("\n>>> [STRATEGY CHANGE] WARM-UP STARTING! (All Stages %100 Dense)")
        if hasattr(model, "module"):
            model.module.set_phase("warmup")
        else:
            model.set_phase("warmup")

    elif epoch == 10:
        logger.info("\n>>> [STRATEGY CHANGE] SEARCH STARTING! (Comeback to layer settings)")
        if hasattr(model, "module"):
            model.module.set_phase("search")
        else:
            model.set_phase("search")

    model.train()

    running_loss = 0.0
    correct1 = 0.0
    correct5 = 0.0
    total = 0

    epoch_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    num_batches = len(trainloader)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Gradient accumulation (if enabled)
        loss = loss / ACCUMULATION_STEPS
        loss.backward()

        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == num_batches:
            # Compute gradient norm for logging
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            current_grad_norm = total_grad_norm ** 0.5

            # Gradient clipping: 5.0 during warmup, 4.0 afterwards
            max_norm = 5.0 if epoch < 10 else 4.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * ACCUMULATION_STEPS

        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
        correct1 += prec1.item()
        correct5 += prec5.item()
        total += labels.size(0)

        # Log every 100 batches
        if i % 100 == 99:
            current_lr = optimizer.param_groups[0]["lr"]
            avg_loss = running_loss / 100.0
            train_acc1 = 100.0 * correct1 / total

            logger.info(
                f"{epoch+1:<6} | {i+1:<6} | "
                f"{avg_loss:<8.4f} | "
                f"{train_acc1:<7.2f}% | "
                f"{current_lr:<9.6f} | "
                f"{current_grad_norm:<10.4f}"
            )

            running_loss = 0.0

    epoch_duration = time.time() - epoch_start
    logger.info(f">>> Epoch {epoch+1} Ended. Duration: {epoch_duration:.2f} sec")

    # ----- Validation -----
    model.eval()

    val_correct1 = 0.0
    val_correct5 = 0.0
    val_total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            p1, p5 = accuracy(outputs, labels, topk=(1, 5))

            val_correct1 += p1.item()
            val_correct5 += p5.item()
            val_total += labels.size(0)

    val_acc1 = 100.0 * val_correct1 / val_total
    val_acc5 = 100.0 * val_correct5 / val_total
    train_acc1_epoch = 100.0 * correct1 / total
    train_acc5_epoch = 100.0 * correct5 / total
    epoch_lr = optimizer.param_groups[0]["lr"]

    generalization_gap = train_acc1_epoch - val_acc1

    logger.info(
        f">>> Epoch {epoch+1} Summary | "
        f"Train Top1: %{train_acc1_epoch:.2f} | Train Top5: %{train_acc5_epoch:.2f} | "
        f"Val Top1: %{val_acc1:.2f} | Val Top5: %{val_acc5:.2f} | "
        f"Gap: %{generalization_gap:.2f} | LR: {epoch_lr:.6f}"
    )

    # Save best model
    if val_acc1 > best_acc:
        best_acc = val_acc1
        best_model_path = os.path.join(SAVE_DIR, "BEST_MODEL.pth")
        logger.info(f"--> NEW RECORD! (Top1: %{best_acc:.2f} | Top5: %{val_acc5:.2f}) Saving...")
        torch.save(model.state_dict(), best_model_path)
    else:
        logger.info(f"    Epoch Val -> Top1: %{val_acc1:.2f} (Best: %{best_acc:.2f}) | Top5: %{val_acc5:.2f}")

    model.train()

    scheduler.step()

    # ----- Cooldown phase (last 30 epochs) -----
    if (epoch + 1) == 170:
        logger.info(">>> [STRATEGY CHANGE] COOLDOWN STARTING! (Explore=0, Exploit=100%)")
        if hasattr(model, "module"):
            model.module.set_phase("cooldown")
        else:
            model.set_phase("cooldown")

    # ----- Periodic topology update during search phase -----
    if epoch >= 10 and (epoch + 1) % 2 == 0 and (epoch + 1) < 170:
        logger.info(f">>> [Topology Update] Mod: SEARCH (Explore) | Epoch {epoch+1}...")

        # Build optimizer momentum map for importance scoring
        optimizer_momentum_map = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    state = optimizer.state[p]
                    if "momentum_buffer" in state:
                        optimizer_momentum_map[id(p)] = state["momentum_buffer"]

        if hasattr(model, "module"):
            if hasattr(model.module, "run_topology_maintenance"):
                model.module.run_topology_maintenance(optimizer_momentum_map)
        else:
            if hasattr(model, "run_topology_maintenance"):
                model.run_topology_maintenance(optimizer_momentum_map)

# ----------------------------------------------------------------------
# Training finished, save last model and report time
# ----------------------------------------------------------------------
total_training_end = time.time()
total_duration = total_training_end - total_training_start
hours = int(total_duration // 3600)
minutes = int((total_duration % 3600) // 60)
seconds = int(total_duration % 60)

logger.info("-" * 25)
logger.info(f"Total Wall Clock Time: {hours}h {minutes}min {seconds}sec")

last_model_name = f"LAST_MODEL_epoch{EPOCHS}.pth"
last_model_path = os.path.join(SAVE_DIR, last_model_name)
torch.save(model.state_dict(), last_model_path)
logger.info(f"Last Model Save: {last_model_path}")

# ----------------------------------------------------------------------
# Final test with best model
# ----------------------------------------------------------------------
best_model_path = os.path.join(SAVE_DIR, "BEST_MODEL.pth")
if os.path.exists(best_model_path):
    logger.info(f"\n>>> Final test with BEST MODEL: ({best_model_path}) loading...")
    model.load_state_dict(torch.load(best_model_path))
else:
    logger.info(f"\n>>> WARNING: BEST MODEL was not found ({best_model_path}), continues with LAST MODEL.")

logger.info("\n--- TEST RESULTS ---")

model.eval()
final_correct1 = 0.0
final_correct5 = 0.0
final_total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        p1, p5 = accuracy(outputs, labels, topk=(1, 5))
        final_correct1 += p1.item()
        final_correct5 += p5.item()
        final_total += labels.size(0)

top1_acc = 100.0 * final_correct1 / final_total
top5_acc = 100.0 * final_correct5 / final_total

logger.info(f"Final Top-1 Accuracy : %{top1_acc:.2f}")
logger.info(f"Final Top-5 Accuracy : %{top5_acc:.2f}")
logger.info("-" * 30)
logger.info(f"Final Top-1 Error    : %{100 - top1_acc:.2f}")
logger.info(f"Final Top-5 Error    : %{100 - top5_acc:.2f}")

# ----------------------------------------------------------------------
# Per-class accuracy
# ----------------------------------------------------------------------
logger.info("\n--- CLASS SPECIFIC RESULTS ---")

classes = testset.classes
num_classes = len(classes)
class_correct = [0.0 for _ in range(num_classes)]
class_total = [0.0 for _ in range(num_classes)]

model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += c[i].item()
            class_total[label] += 1

logger.info(f"{'ID':<3} | {'CLASS':<20} | {'ACCURACY':<10}")
logger.info("-" * 40)
for i in range(num_classes):
    if class_total[i] > 0:
        acc = 100.0 * class_correct[i] / class_total[i]
        logger.info(f"{i:<3} | {classes[i]:<20} | %{acc:.2f}")
    else:
        logger.info(f"{i:<3} | {classes[i]:<20} | No Data")

logger.info(f"\nLog saved: {os.path.abspath(log_filename)}")
