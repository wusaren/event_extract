# PLMEE 项目运行指南

## 环境准备

1. 安装Python 3.8+
2. 安装依赖库：
```bash
pip install torch transformers tqdm
```

## 训练模型

```bash
python src/main.py \
  --model_name bert-base-chinese \
  --train_file data/DuEE/train.json \
  --eval_file data/DuEE/dev.json \
  --num_epochs 10 \
  --batch_size 32
```

参数说明：
- `--model_name`: 预训练模型名称 (如: bert-base-chinese)
- `--train_file`: 训练数据路径
- `--eval_file`: 验证数据路径 (可选)
- `--num_epochs`: 训练轮数
- `--batch_size`: 批大小

## 预测

1. 准备预测脚本 `predict.py` (基于重构后的代码)
2. 运行预测：
```bash
python src/predict.py \
  --model_path saved_models/best_model.pt \
  --test_file data/DuEE/test.json \
  --output_file results/predictions.json
```

## 数据格式

训练数据应为JSON格式，示例：
```json
{
  "text": "文本内容",
  "event_types": [事件类型ID],
  "trigger_spans": [[start, end], ...],
  "argument_spans": [[start, end, role_id], ...]
}
```

## 目录结构
```
PLMEE/
├── data/               # 数据目录
├── src/                # 源代码
│   ├── data_processor.py  # 数据处理
│   ├── model.py        # 模型定义
│   ├── trainer.py      # 训练逻辑
│   └── main.py         # 主程序
├── saved_models/       # 模型保存目录
└── results/            # 预测结果
