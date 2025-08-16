import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class EventDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self.load_data(data_path)
        
    def load_data(self, data_path):
        # 实现原始数据加载逻辑
        # 返回包含文本、触发词、论元等信息的列表
        pass
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 使用transformers tokenizer处理文本
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理标签
        labels = {
            'trigger_spans': torch.tensor(item['trigger_spans']),
            'argument_spans': torch.tensor(item['argument_spans']),
            'event_types': torch.tensor(item['event_types'])
        }
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels
        }

def create_dataloader(dataset, batch_size=32, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: x  # 可能需要自定义collate_fn
    )
