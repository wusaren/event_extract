import torch
import torch.nn as nn
from transformers import AutoModel

class TriggerExtractor(nn.Module):
    def __init__(self, model_name, num_event_types):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.event_classifier = nn.Linear(self.bert.config.hidden_size, num_event_types)
        self.span_classifier = nn.Linear(self.bert.config.hidden_size, 2)  # start/end
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 事件类型分类
        event_logits = self.event_classifier(sequence_output[:, 0, :])  # [CLS] token
        
        # 触发词span预测
        span_logits = self.span_classifier(sequence_output)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        
        return {
            'event_logits': event_logits,
            'start_logits': start_logits.squeeze(-1),
            'end_logits': end_logits.squeeze(-1)
        }

class ArgumentExtractor(nn.Module):
    def __init__(self, model_name, num_roles):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.role_classifier = nn.Linear(self.bert.config.hidden_size, num_roles)
        self.span_classifier = nn.Linear(self.bert.config.hidden_size, 2)  # start/end
        
    def forward(self, input_ids, attention_mask, event_types):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 论元角色分类
        role_logits = self.role_classifier(sequence_output)
        
        # 论元span预测
        span_logits = self.span_classifier(sequence_output)
        start_logits, end_logits = span_logits.split(1, dim=-1)
        
        return {
            'role_logits': role_logits,
            'start_logits': start_logits.squeeze(-1),
            'end_logits': end_logits.squeeze(-1)
        }

class EventExtractor(nn.Module):
    def __init__(self, model_name, num_event_types, num_roles):
        super().__init__()
        self.trigger_extractor = TriggerExtractor(model_name, num_event_types)
        self.argument_extractor = ArgumentExtractor(model_name, num_roles)
        
    def forward(self, input_ids, attention_mask):
        trigger_outputs = self.trigger_extractor(input_ids, attention_mask)
        argument_outputs = self.argument_extractor(input_ids, attention_mask, 
                                                 trigger_outputs['event_logits'].argmax(-1))
        return {
            **trigger_outputs,
            **argument_outputs
        }
