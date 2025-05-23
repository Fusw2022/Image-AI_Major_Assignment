import sys
import os
import torch
import torch.nn as nn
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QComboBox, QSpinBox,
                             QDoubleSpinBox, QTextEdit, QTabWidget, QGroupBox, QCheckBox,
                             QProgressBar, QMessageBox, QSplitter, QListWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from cnn_model import CNNModel1, CNNModel2, CNNModel3,MediumCNN,EnhancedCNN
from data_loader import get_data_loaders
from train_evaluate import train_model, evaluate_model, plot_roc_curve, TrainingSignal
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
batch_size=32
img_size=224
lr=0.0001
class TrainingThread(QThread):
    update_progress = pyqtSignal(int, str)
    training_complete = pyqtSignal(object, object)
    training_error = pyqtSignal(str)
    update_log = pyqtSignal(str)
    start_comparison = pyqtSignal()  # 新增：转发开始比较信号
    end_comparison = pyqtSignal()    # 新增：转发结束比较信号

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, epochs, device):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device
        self.signal = TrainingSignal()
        self.signal.update_progress.connect(self.update_progress)
        self.signal.update_log.connect(self.update_log)
        self.signal.start_comparison.connect(self.start_comparison)  # 连接信号
        self.signal.end_comparison.connect(self.end_comparison)    # 连接信号

    def run(self):
        try:
            model, history = train_model(
                self.model, self.train_loader, self.val_loader,
                self.criterion, self.optimizer, self.epochs, self.device, self.signal
            )
            self.training_complete.emit(model, history)
        except Exception as e:
            self.training_error.emit(str(e))

class EvaluationThread(QThread):
    evaluation_complete = pyqtSignal(object, object, object, object)
    evaluation_error = pyqtSignal(str)

    def __init__(self, model, test_loader, device, num_classes):
        super().__init__()
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes

    def run(self):
        try:
            metrics, roc_data, labels, preds = evaluate_model(
                self.model, self.test_loader, self.device, self.num_classes
            )
            self.evaluation_complete.emit(metrics, roc_data, labels, preds)
        except Exception as e:
            self.evaluation_error.emit(str(e))

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.fig.tight_layout()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("中药图片分类系统")
        self.setGeometry(100, 100, 1200, 800)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = []
        self.metrics = None
        self.roc_data = None
        self.training_in_progress = False  # 新增标志，用于避免重复训练

        self.init_ui()

    def init_ui(self):
        # 创建主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 创建左侧控制面板
        control_panel = QVBoxLayout()

        # 数据集加载部分
        dataset_group = QGroupBox("数据集设置")
        dataset_layout = QVBoxLayout()

        self.dataset_path_label = QLabel("未选择数据集路径")
        dataset_layout.addWidget(self.dataset_path_label)

        browse_btn = QPushButton("浏览数据集")
        browse_btn.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(browse_btn)

        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(64, 512)
        self.img_size_spin.setValue(img_size)
        self.img_size_spin.setSuffix(" 像素")
        img_size_layout = QHBoxLayout()
        img_size_layout.addWidget(QLabel("图像大小:"))
        img_size_layout.addWidget(self.img_size_spin)
        dataset_layout.addLayout(img_size_layout)

        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(4, 128)
        self.batch_size_spin.setValue(batch_size)
        self.batch_size_spin.setSuffix(" 样本")
        batch_size_layout = QHBoxLayout()
        batch_size_layout.addWidget(QLabel("批次大小:"))
        batch_size_layout.addWidget(self.batch_size_spin)
        dataset_layout.addLayout(batch_size_layout)

        self.augment_checkbox = QCheckBox("数据增强")
        self.augment_checkbox.setChecked(True)
        dataset_layout.addWidget(self.augment_checkbox)

        load_dataset_btn = QPushButton("加载数据集")
        load_dataset_btn.clicked.connect(self.load_dataset)
        dataset_layout.addWidget(load_dataset_btn)

        self.dataset_status = QLabel("状态: 未加载")
        dataset_layout.addWidget(self.dataset_status)

        dataset_group.setLayout(dataset_layout)
        control_panel.addWidget(dataset_group)

        # 模型设置部分
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()

        self.model_combobox = QComboBox()
        self.model_combobox.addItems(["MediumCNN","EnhancedCNN","Model1", "Model2", "Model3"])
        model_layout.addWidget(QLabel("选择模型架构:"))
        model_layout.addWidget(self.model_combobox)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1)
        self.lr_spin.setValue(lr)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setDecimals(4)
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("学习率:"))
        lr_layout.addWidget(self.lr_spin)
        model_layout.addLayout(lr_layout)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100)
        self.epochs_spin.setValue(10)
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("训练轮次:"))
        epochs_layout.addWidget(self.epochs_spin)
        model_layout.addLayout(epochs_layout)

        init_model_btn = QPushButton("初始化模型")
        init_model_btn.clicked.connect(self.init_model)
        model_layout.addWidget(init_model_btn)

        self.model_status = QLabel("状态: 未初始化")
        model_layout.addWidget(self.model_status)

        model_group.setLayout(model_layout)
        control_panel.addWidget(model_group)

        # 训练部分
        train_group = QGroupBox("训练")
        train_layout = QVBoxLayout()

        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        train_layout.addWidget(self.train_progress)

        self.train_status = QLabel("状态: 就绪")
        train_layout.addWidget(self.train_status)

        self.train_log_text = QTextEdit()  # 新增文本框，用于显示训练日志
        self.train_log_text.setReadOnly(True)
        train_layout.addWidget(self.train_log_text)

        train_btn = QPushButton("开始训练")
        train_btn.clicked.connect(self.start_training)
        train_layout.addWidget(train_btn)

        train_group.setLayout(train_layout)
        control_panel.addWidget(train_group)

        # 评估部分
        eval_group = QGroupBox("评估")
        eval_layout = QVBoxLayout()

        eval_btn = QPushButton("评估模型")
        eval_btn.clicked.connect(self.evaluate_model)
        eval_layout.addWidget(eval_btn)

        self.eval_status = QLabel("状态: 就绪")
        eval_layout.addWidget(self.eval_status)

        eval_group.setLayout(eval_layout)
        control_panel.addWidget(eval_group)

        control_panel.addStretch(1)

        # 创建右侧显示面板
        display_panel = QVBoxLayout()

        self.tab_widget = QTabWidget()

        # 训练历史标签页
        self.history_tab = QWidget()
        self.history_layout = QVBoxLayout(self.history_tab)
        self.history_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.history_layout.addWidget(self.history_canvas)
        self.tab_widget.addTab(self.history_tab, "训练历史")

        # 评估指标标签页
        self.metrics_tab = QWidget()
        self.metrics_layout = QVBoxLayout(self.metrics_tab)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_layout.addWidget(self.metrics_text)
        self.tab_widget.addTab(self.metrics_tab, "评估指标")

        # ROC曲线标签页
        self.roc_tab = QWidget()
        self.roc_layout = QVBoxLayout(self.roc_tab)
        self.roc_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.roc_layout.addWidget(self.roc_canvas)
        self.tab_widget.addTab(self.roc_tab, "ROC曲线")

        # 预测结果标签页
        self.prediction_tab = QWidget()
        self.prediction_layout = QVBoxLayout(self.prediction_tab)

        self.sample_list = QListWidget()
        self.sample_list.itemClicked.connect(self.show_prediction_details)
        self.prediction_layout.addWidget(QLabel("测试样本:"))
        self.prediction_layout.addWidget(self.sample_list)

        self.prediction_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.prediction_layout.addWidget(self.prediction_canvas)

        self.tab_widget.addTab(self.prediction_tab, "预测结果")

        display_panel.addWidget(self.tab_widget)

        # 将左右面板添加到主布局
        left_widget = QWidget()
        left_widget.setLayout(control_panel)
        right_widget = QWidget()
        right_widget.setLayout(display_panel)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter)

    def browse_dataset(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if dir_path:
            self.dataset_path = dir_path
            self.dataset_path_label.setText(dir_path)

    def load_dataset(self):
        if not hasattr(self, 'dataset_path'):
            QMessageBox.warning(self, "警告", "请先选择数据集路径!")
            return

        try:
            self.dataset_status.setText("状态: 加载中...")
            QApplication.processEvents()
            img_size = self.img_size_spin.value()
            batch_size = self.batch_size_spin.value()
            augment = self.augment_checkbox.isChecked()

            self.train_loader, self.val_loader, self.test_loader, self.class_names = get_data_loaders(
                self.dataset_path, batch_size=batch_size, img_size=img_size, augment=augment
            )

            self.dataset_status.setText(f"状态: 加载成功! 类别数: {len(self.class_names)}, 训练样本: {len(self.train_loader.dataset)}, 验证样本: {len(self.val_loader.dataset)}, 测试样本: {len(self.test_loader.dataset)}")

            # 更新样本列表
            self.sample_list.clear()
            for i in range(len(self.test_loader.dataset)):
                img_path, true_label = self.test_loader.dataset.samples[i]
                class_name = self.class_names[true_label]
                self.sample_list.addItem(f"样本 {i+1}: {os.path.basename(img_path)} (真实类别: {class_name})")

        except Exception as e:
            self.dataset_status.setText(f"状态: 加载失败 - {str(e)}")
            QMessageBox.critical(self, "错误", f"加载数据集时出错: {str(e)}")

    def init_model(self):
        if not self.class_names:
            QMessageBox.warning(self, "警告", "请先加载数据集!")
            return

        try:
            self.model_status.setText("状态: 初始化中...")
            QApplication.processEvents()

            model_type = self.model_combobox.currentText()
            num_classes = len(self.class_names)

            if model_type == "Model1":
                self.model = CNNModel1(num_classes,self.img_size_spin.value())
            elif model_type == "Model2":
                self.model = CNNModel2(num_classes,self.img_size_spin.value())
            elif model_type == "Model3":
                self.model = CNNModel3(num_classes,self.img_size_spin.value())
            elif model_type == "MediumCNN":
                self.model = MediumCNN(num_classes,self.img_size_spin.value())
            elif model_type == "EnhancedCNN":
                self.model = EnhancedCNN(num_classes,self.img_size_spin.value())

            self.model = self.model.to(self.device)
            self.model_status.setText(f"状态: {model_type} 初始化成功!")

        except Exception as e:
            self.model_status.setText(f"状态: 初始化失败 - {str(e)}")
            QMessageBox.critical(self, "错误", f"初始化模型时出错: {str(e)}")

    def start_training(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先初始化模型!")
            return

        if self.train_loader is None or self.val_loader is None:
            QMessageBox.warning(self, "警告", "请先加载数据集!")
            return

        try:
            lr = self.lr_spin.value()
            epochs = self.epochs_spin.value()

            # 重置进度条
            self.train_progress.setValue(0)
            self.train_status.setText("状态: 准备训练...")

            # 创建训练线程
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

            self.training_thread = TrainingThread(
                self.model, self.train_loader, self.val_loader,
                criterion, optimizer, epochs, self.device
            )

            # 连接信号
            self.training_thread.update_progress.connect(self.update_training_progress)
            self.training_thread.training_complete.connect(self.training_complete)
            self.training_thread.training_error.connect(self.training_error)
            # 新增：连接日志信号
            self.training_thread.signal.update_log.connect(self.show_log)
            # 连接新增信号
            self.training_thread.start_comparison.connect(self.show_comparison_status)
            self.training_thread.end_comparison.connect(self.resume_training_status)

            # 启动线程
            self.training_thread.start()

        except Exception as e:
            self.train_status.setText(f"状态: 训练失败 - {str(e)}")
            QMessageBox.critical(self, "错误", f"开始训练时出错: {str(e)}")

    def update_training_progress(self, progress, message):
        self.train_progress.setValue(progress)
        self.train_status.setText(f"状态: {message}")

    def show_comparison_status(self):
        """显示比较分析状态"""
        self.train_status.setText("状态: 正在验证模型...")

    def resume_training_status(self):
        """恢复训练状态"""
        self.train_status.setText(f"状态: 正在准备下一次训练...")

    def training_complete(self, model, history):
        self.model = model
        self.train_status.setText("状态: 训练完成!")
        self.train_progress.setValue(100)
        self.training_in_progress = False  # 标记训练完成

        # 显示训练完成消息
        QMessageBox.information(self, "训练完成", "模型训练已完成!")

        # 绘制训练历史
        self.history_canvas.fig.clear()  # 清空整个画布

        # 创建 1 行 2 列的子图布局
        ax1 = self.history_canvas.fig.add_subplot(1, 2, 1)  # 第一个子图用于绘制损失
        ax2 = self.history_canvas.fig.add_subplot(1, 2, 2)  # 第二个子图用于绘制准确率

        epochs = range(1, len(history['train_loss']) + 1)

        # 在第一个子图上绘制损失曲线
        ax1.plot(epochs, history['train_loss'], 'b-', label='训练损失')
        ax1.plot(epochs, history['val_loss'], 'r-', label='验证损失')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.legend()

        # 在第二个子图上绘制准确率曲线
        ax2.plot(epochs, history['train_acc'], 'b-', label='训练准确率')
        ax2.plot(epochs, history['val_acc'], 'r-', label='验证准确率')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()

        # 调整子图布局
        self.history_canvas.fig.tight_layout()

        # 重新绘制画布
        self.history_canvas.draw()

    def training_error(self, error_msg):
        self.train_status.setText(f"状态: 训练失败 - {error_msg}")
        QMessageBox.critical(self, "训练错误", f"训练过程中出错: {error_msg}")
        self.training_in_progress = False  # 标记训练失败

    def update_epoch_info(self, epoch, epochs, train_loss, train_acc, val_loss, val_acc):
        # 输出训练结果信息到文本框
        log_text = f'Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n'
        self.show_log(log_text)

    def show_log(self, log_text):
        self.train_log_text.append(log_text)

    def evaluate_model(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先初始化并训练模型!")
            return

        if self.test_loader is None:
            QMessageBox.warning(self, "警告", "请先加载数据集!")
            return

        try:
            self.eval_status.setText("状态: 评估中...")

            self.evaluation_thread = EvaluationThread(
                self.model, self.test_loader, self.device, len(self.class_names)
            )
            self.evaluation_thread.evaluation_complete.connect(self.evaluation_complete)
            self.evaluation_thread.evaluation_error.connect(self.evaluation_error)
            self.evaluation_thread.start()

        except Exception as e:
            self.eval_status.setText(f"状态: 评估失败 - {str(e)}")
            QMessageBox.critical(self, "错误", f"评估模型时出错: {str(e)}")

    def evaluation_complete(self, metrics, roc_data, labels, preds):
        self.metrics = metrics
        self.roc_data = roc_data

        self.eval_status.setText("状态: 评估完成!")

        # 显示评估指标
        metrics_text = f"准确率: {metrics['accuracy']:.4f}\n"
        metrics_text += f"精确率: {metrics['precision']:.4f}\n"
        metrics_text += f"召回率: {metrics['recall']:.4f}\n"
        metrics_text += f"F1分数: {metrics['f1']:.4f}\n"

        self.metrics_text.setText(metrics_text)

        # 彻底重置ROC画布
        self.reset_canvas(self.roc_canvas)

        # 绘制ROC曲线
        fig = plot_roc_curve(roc_data, self.class_names)

        # 正确复制新Figure的内容到roc_canvas
        self.copy_figure_contents(fig, self.roc_canvas)

        # 调整布局并绘制
        self.roc_canvas.fig.tight_layout()
        self.roc_canvas.draw()

        QMessageBox.information(self, "评估完成", "模型评估已完成!")

    def reset_canvas(self, canvas):
        """彻底重置画布，清除所有内容和状态"""
        canvas.fig.clear()
        canvas.axes = canvas.fig.add_subplot(111)
        # 重置常用属性
        canvas.axes.set_xlim([0.0, 1.0])
        canvas.axes.set_ylim([0.0, 1.05])
        canvas.axes.set_xlabel('假正例率')
        canvas.axes.set_ylabel('真正例率')
        canvas.axes.set_title('ROC曲线')
        canvas.axes.legend(loc="lower right", fontsize=10)
        canvas.axes.grid(True)

    def copy_figure_contents(self, source_fig, target_canvas):
        """将一个Figure的内容复制到另一个Canvas的Figure"""
        # 清除目标画布
        target_canvas.fig.clear()
        target_canvas.axes = target_canvas.fig.add_subplot(111)

        # 复制所有线条、文本和其他元素
        for source_ax in source_fig.axes:
            # 复制线条
            for line in source_ax.lines:
                x, y = line.get_xdata(), line.get_ydata()
                label = line.get_label()
                color = line.get_color()
                linestyle = line.get_linestyle()
                target_canvas.axes.plot(x, y, label=label, color=color, linestyle=linestyle)

            # 复制文本（如标题、标签）
            if source_ax.get_title():
                target_canvas.axes.set_title(source_ax.get_title())
            if source_ax.get_xlabel():
                target_canvas.axes.set_xlabel(source_ax.get_xlabel())
            if source_ax.get_ylabel():
                target_canvas.axes.set_ylabel(source_ax.get_ylabel())

            # 复制图例
            if source_ax.get_legend():
                target_canvas.axes.legend()

    def evaluation_error(self, error_msg):
        self.eval_status.setText(f"状态: 评估失败 - {error_msg}")
        QMessageBox.critical(self, "评估错误", f"评估过程中出错: {error_msg}")

    def show_prediction_details(self, item):
        if not hasattr(self, 'test_loader') or self.model is None:
            return

        try:
            sample_idx = self.sample_list.row(item)
            image, true_label = self.test_loader.dataset[sample_idx]
            image_path, _ = self.test_loader.dataset.samples[sample_idx]

            # 预测
            self.model.eval()
            with torch.no_grad():
                image_tensor = image.unsqueeze(0).to(self.device)
                outputs = self.model(image_tensor)

                # 处理二分类和多分类的概率输出
                num_classes = len(self.class_names)
                if num_classes == 2:
                    # 二分类：概率为正类的单个值（形状为 [1]）
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                else:
                    # 多分类：概率为所有类别的数组（形状为 [num_classes]）
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]  # 取第一个样本的概率

                pred_prob = np.max(probs)
                pred_label = np.argmax(probs)

            # 清空画布
            self.prediction_canvas.fig.clear()

            # 转换图像以便显示
            img_np = image.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            # 绘制图像和预测结果
            self.prediction_canvas.axes1 = self.prediction_canvas.fig.add_subplot(131)
            self.prediction_canvas.axes1.imshow(img_np)
            self.prediction_canvas.axes1.set_title(
                f"真实类别: {self.class_names[true_label]}\n"
                f"预测类别: {self.class_names[pred_label]} (置信度: {pred_prob:.4f})",
                fontsize=10
            )
            self.prediction_canvas.axes.axis('off')
            self.prediction_canvas.axes1.axis('off')

            # 绘制水平条形图（关键修复点）
            self.prediction_canvas.fig.subplots_adjust(left=0.3)  # 调整边距避免标签被截断
            self.prediction_canvas.axes2 = self.prediction_canvas.fig.add_subplot(133)

            # 处理类别名称和概率
            classes = self.class_names
            y_pos = np.arange(len(classes))
            # 确保probs是一维数组（多分类）或扩展为二维（二分类）
            if num_classes == 2:
                probs = np.array([1 - probs[0], probs[0]])  # 补全负类概率，形状为 [2]
            self.prediction_canvas.axes2.barh(
                y_pos, probs, align='center', color=['#6c757d', '#4CAF50']  # 灰（负类）绿（正类）
            )
            self.prediction_canvas.axes2.set_yticks(y_pos)
            self.prediction_canvas.axes2.set_yticklabels(classes, fontsize=8)
            self.prediction_canvas.axes2.invert_yaxis()  # 使类别从上到下排列
            self.prediction_canvas.axes2.set_xlabel('概率', fontsize=8)
            self.prediction_canvas.axes2.set_title('类别预测概率分布', fontsize=10)
            self.prediction_canvas.axes2.grid(axis='x', linestyle='--', alpha=0.7)

            self.prediction_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"显示预测结果时出错: {str(e)}")
            print(f"错误详情: {e}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()