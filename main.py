import argparse

def main():
    print("若要使用图形界面，请直接运行gui_interface.py，或在命令行中运行如下命令：python main.py --gui")
    parser = argparse.ArgumentParser(description='中药图片分类系统')
    parser.add_argument('--data_dir', type=str, default='中药数据集', help='数据集目录路径')
    parser.add_argument('--model_type', type=str, default='Model1',
                        choices=['Model1', 'Model2', 'Model3', 'MediumCNN', 'EnhancedCNN'], help='模型类型')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--img_size', type=int, default=224, help='图像大小')
    parser.add_argument('--no_augment', action='store_true', help='不使用数据增强')
    parser.add_argument('--gui', action='store_true', help='启动图形界面')
    parser.add_argument('--epsilon', type=float, default=0.01, help='FGSM 扰动强度')
    parser.add_argument('--use_ssl', action='store_true', help='使用自监督学习')

    args = parser.parse_args()

    if args.gui:
        from gui_interface import main as gui_main
        gui_main()
    else:
        # 命令行模式
        if not args.data_dir:
            print("错误: 使用命令行模式时必须指定数据集目录 (--data_dir)")
            return

        import torch
        import torch.nn as nn
        import torch.optim as optim
        from cnn_model import CNNModel1, CNNModel2, CNNModel3, MediumCNN, EnhancedCNN
        from data_loader import get_data_loaders
        from train_evaluate import train_model, evaluate_model, plot_roc_curve, TrainingSignal
        import matplotlib.pyplot as plt

        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {device}")

        img_size = args.img_size

        # 加载数据
        print("加载数据集...")
        train_loader, val_loader, test_loader, class_names = get_data_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            img_size=img_size,
            augment=not args.no_augment
        )
        print(f"数据集加载完成! 类别数: {len(class_names)}")

        # 初始化模型
        print(f"初始化 {args.model_type} 模型...")
        if args.model_type == "Model1":
            model = CNNModel1(len(class_names), img_size=img_size)
        elif args.model_type == "Model2":
            model = CNNModel2(len(class_names), img_size=img_size)
        elif args.model_type == "Model3":
            model = CNNModel3(len(class_names), img_size=img_size)
        elif args.model_type == "MediumCNN":
            model = MediumCNN(len(class_names), img_size=img_size)
        elif args.model_type == "EnhancedCNN":
            model = EnhancedCNN(len(class_names), img_size=img_size)

        model = model.to(device)

        # 训练模型
        print(f"开始训练模型 (学习率: {args.lr}, 轮次: {args.epochs})...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        signal = TrainingSignal()

        def update_progress(progress, message):
            print(f"进度: {progress}%, {message}")

        def show_log(log_text):
            print(log_text)

        signal.update_progress.connect(update_progress)
        signal.update_log.connect(show_log)
        try:
            model, history = train_model(model, train_loader, val_loader, criterion, optimizer, args.epochs, device, signal, args.epsilon, args.use_ssl)
        except:
            # 检查输出和标签的批次大小是否一致
            print(f"警告: 输出批次大小与标签批次大小不匹配")
            return

        # 评估模型
        print("评估模型...")
        metrics, roc_data, _, _ = evaluate_model(model, test_loader, device, len(class_names))

        # 打印评估结果
        print("\n=== 评估结果 ===")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")

        # 绘制训练历史和ROC曲线
        plt.figure(figsize=(12, 5))

        plt.subplot(121)
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], 'b-', label='训练损失')
        plt.plot(epochs, history['val_loss'], 'r-', label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.title('训练和验证损失')

        plt.subplot(122)
        plt.plot(epochs, history['train_acc'], 'b-', label='训练准确率')
        plt.plot(epochs, history['val_acc'], 'r-', label='验证准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率 (%)')
        plt.legend()
        plt.title('训练和验证准确率')

        plt.tight_layout()
        plt.savefig('training_history.png')

        plt.figure(figsize=(10, 8))
        plot_roc_curve(roc_data, class_names)
        plt.savefig('roc_curve.png')

        print("\n训练历史和ROC曲线已保存为图片文件.")


if __name__ == "__main__":
    main()    