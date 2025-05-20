import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 定义一个信号类，用于传递训练进度和批次信息
from PyQt5.QtCore import pyqtSignal, QObject


class TrainingSignal(QObject):
    update_progress = pyqtSignal(int, str)
    update_log = pyqtSignal(str)  # 新增信号，用于发送训练日志信息
    start_comparison = pyqtSignal()  # 新增：开始比较信号
    end_comparison = pyqtSignal()  # 新增：结束比较信号


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, signal):
    best_val_loss = float('inf')
    best_model = None
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    total_batches = len(train_loader) * epochs
    current_batch = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            current_batch += 1
            progress = int((current_batch / total_batches) * 100)
            message = f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}'
            signal.update_progress.emit(progress, message)

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        # **新增：发送开始比较信号**
        signal.start_comparison.emit()

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        log_text = f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
        print(log_text)
        signal.update_log.emit(log_text)  # 发送训练日志信息

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            print(f'Saved best model at epoch {epoch + 1}')
        # **新增：发送结束比较信号**
        signal.end_comparison.emit()

    model.load_state_dict(best_model)
    return model, history

def evaluate_model(model, test_loader, device, num_classes):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    num_batches = len(test_loader)

    with tqdm(total=num_batches, desc='Evaluation', unit='batch') as pbar:
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # 处理二分类和多分类情况
                if num_classes == 2:
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # 只取正类概率
                    _, preds = torch.max(outputs, 1)
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs)

                pbar.set_postfix({'Batch': batch_idx + 1})
                pbar.update(1)

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 计算ROC曲线
    if num_classes == 2:
        # 二分类情况
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc
        }
    else:
        # 多分类情况
        y_test_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        y_score_bin = np.array(all_probs)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score_bin.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        roc_data = {
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'num_classes': num_classes
        }

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    return metrics, roc_data, all_labels, all_preds


def plot_roc_curve(roc_data, class_names=None):
    plt.figure(figsize=(10, 8))

    if 'num_classes' in roc_data and roc_data['num_classes'] > 2:
        # 绘制多类别ROC曲线
        plt.plot(roc_data['fpr']['micro'], roc_data['tpr']['micro'],
                 label=f'micro-average ROC curve (area = {roc_data["roc_auc"]["micro"]:0.2f})',
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(roc_data['fpr']['macro'], roc_data['tpr']['macro'],
                 label=f'macro-average ROC curve (area = {roc_data["roc_auc"]["macro"]:0.2f})',
                 color='navy', linestyle=':', linewidth=4)

        for i in range(roc_data['num_classes']):
            label = class_names[i] if class_names and i < len(class_names) else f'Class {i}'
            plt.plot(roc_data['fpr'][i], roc_data['tpr'][i], lw=2,
                     label=f'ROC curve of class {label} (area = {roc_data["roc_auc"][i]:0.2f})')
    else:
        # 绘制二分类ROC曲线
        plt.plot(roc_data['fpr'], roc_data['tpr'], color='darkorange',
                 lw=2, label=f'ROC curve (area = {roc_data["roc_auc"]:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()

    return plt.gcf()