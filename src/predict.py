import argparse
import json
import torch
from transformers import AutoTokenizer
from model import EventExtractor
from data_processor import EventDataset

def load_model(model_path, model_name, num_event_types, num_roles):
    """加载训练好的模型"""
    model = EventExtractor(model_name, num_event_types, num_roles)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model, tokenizer, text, max_length=512):
    """单条文本预测"""
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
    
    # 解析预测结果
    event_types = torch.argmax(outputs['event_logits'], dim=-1).tolist()
    trigger_spans = extract_spans(outputs['start_logits'], outputs['end_logits'])
    argument_spans = extract_spans(outputs['role_start_logits'], outputs['role_end_logits'])
    
    return {
        'text': text,
        'event_types': event_types,
        'trigger_spans': trigger_spans,
        'argument_spans': argument_spans
    }

def extract_spans(start_logits, end_logits, threshold=0.5):
    """从logits中提取span"""
    spans = []
    start_probs = torch.sigmoid(start_logits)
    end_probs = torch.sigmoid(end_logits)
    
    for i in range(len(start_probs)):
        if start_probs[i] > threshold and end_probs[i] > threshold:
            spans.append((i, i))  # 简化为单token span
    return spans

def main():
    parser = argparse.ArgumentParser(description="PLMEE 事件抽取预测脚本")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--model_name", type=str, default="bert-base-chinese", help="预训练模型名称")
    parser.add_argument("--test_file", type=str, required=True, help="测试数据文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="预测结果输出路径")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    args = parser.parse_args()

    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 加载数据获取类别数
    dataset = EventDataset(args.test_file, args.model_name, args.max_length)
    
    # 加载模型
    model = load_model(
        args.model_path,
        args.model_name,
        dataset.num_event_types,
        dataset.num_roles
    ).to(device)
    
    # 预测
    results = []
    with open(args.test_file) as f:
        for line in f:
            data = json.loads(line)
            result = predict(model, tokenizer, data['text'], args.max_length)
            results.append(result)
    
    # 保存结果
    with open(args.output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"预测完成，结果已保存到 {args.output_file}")

if __name__ == "__main__":
    main()
