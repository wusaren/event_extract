import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

class EventExtractorTrainer:
    def __init__(self, model, device, learning_rate=5e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            inputs = {
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            
            # 计算损失
            loss = self.compute_loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def compute_loss(self, outputs, labels):
        # 触发词分类损失
        trigger_loss = F.cross_entropy(outputs['event_logits'], labels['event_types'])
        
        # 论元角色分类损失
        argument_loss = F.cross_entropy(outputs['role_logits'], labels['argument_roles'])
        
        # Span预测损失
        start_loss = F.binary_cross_entropy_with_logits(
            outputs['start_logits'], 
            labels['trigger_spans'][:, :, 0].float()
        )
        end_loss = F.binary_cross_entropy_with_logits(
            outputs['end_logits'],
            labels['trigger_spans'][:, :, 1].float()
        )
        
        return trigger_loss + argument_loss + start_loss + end_loss

    def evaluate(self, eval_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device)
                }
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(**inputs)
                loss = self.compute_loss(outputs, labels)
                total_loss += loss.item()
                
        return total_loss / len(eval_loader)

    def predict(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device)
            )
        return outputs
