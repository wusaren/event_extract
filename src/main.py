import argparse
import os
import torch
from data_processor import EventDataset, create_dataloader
from model import EventExtractor
from trainer import EventExtractorTrainer

def main():
    parser = argparse.ArgumentParser(description="PLMEE 事件抽取模型训练脚本")
    
    # 数据参数
    parser.add_argument("--train_file", type=str, required=True, 
                      help="训练数据文件路径")
    parser.add_argument("--eval_file", type=str, default=None,
                      help="验证数据文件路径")
    parser.add_argument("--test_file", type=str, default=None,
                      help="测试数据文件路径")
    
    # 模型参数
    parser.add_argument("--model_name", type=str, default="bert-base-chinese",
                      help="预训练模型名称")
    parser.add_argument("--max_length", type=int, default=512,
                      help="最大序列长度")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32,
                      help="训练批大小")
    parser.add_argument("--num_epochs", type=int, default=10,
                      help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                      help="学习率")
    parser.add_argument("--output_dir", type=str, default="./saved_models",
                      help="模型保存目录")
    
    args = parser.parse_args()
    
    # 初始化设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("Loading datasets...")
    train_dataset = EventDataset(args.train_file, args.model_name, args.max_length)
    train_loader = create_dataloader(train_dataset, args.batch_size, shuffle=True)
    
    if args.eval_file:
        eval_dataset = EventDataset(args.eval_file, args.model_name, args.max_length)
        eval_loader = create_dataloader(eval_dataset, args.batch_size, shuffle=False)
    
    # 初始化模型
    print("Initializing model...")
    model = EventExtractor(
        args.model_name,
        num_event_types=train_dataset.num_event_types,
        num_roles=train_dataset.num_roles
    )
    model.to(device)
    
    # 初始化训练器
    trainer = EventExtractorTrainer(model, device, args.learning_rate)
    
    # 训练循环
    print("Starting training...")
    best_loss = float('inf')
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # 训练
        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # 验证
        if args.eval_file:
            eval_loss = trainer.evaluate(eval_loader)
            print(f"Eval Loss: {eval_loss:.4f}")
            
            # 保存最佳模型
            if eval_loss < best_loss:
                best_loss = eval_loss
                model_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save(model.state_dict(), model_path)
                print(f"Saved best model to {model_path}")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
