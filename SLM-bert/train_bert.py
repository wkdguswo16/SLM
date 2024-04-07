from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer
from models.config import MultiHeadConfig
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from models.slm import ScalableLM
from transformers.models.bert import BertForSequenceClassification
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Union, List, Any, Dict
from dataset.utils import get_dataset_by_name
from datasets.utils.logging import disable_progress_bar
import os
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TASK_TO_DATA_NAME = {
    'dbpedia': 'dbpedia',
    'amazon': 'amazon',
    'yelp': 'yelp',
    'yahoo': 'yahoo',
    'ag_news': 'ag'
}

TASK_TO_NUM_LABELS = {
    'yelp': 5,
    'ag_news': 4,
    'yahoo': 10,
    'dbpedia': 14,
    'amazon': 5,
}

TASK_LIST = {
    "1": ['ag_news', 'yelp', 'yahoo'],
    "2": ['yelp', 'yahoo', 'ag_news'],
    "3": ['yahoo', 'ag_news', 'yelp'],
    "4": ['ag_news', 'yelp', 'amazon', 'yahoo', 'dbpedia'],
    "4-4": ['ag_news', 'yelp', 'amazon', 'yahoo'],
    "4-3": ['ag_news', 'yelp', 'amazon'],
    "4-2": ['ag_news', 'yelp'],
    "4-1": ['ag_news'],
    "5": ['yelp', 'yahoo', 'amazon', 'dbpedia', 'ag_news'],
    "5-4": ['yelp', 'yahoo', 'amazon', 'dbpedia'],
    "5-3": ['yelp', 'yahoo', 'amazon'],
    "5-2": ['yelp', 'yahoo'],
    "5-1": ['yelp'],
    "6": ['dbpedia', 'yahoo', 'ag_news', 'amazon', 'yelp'],
    "6-4": ['dbpedia', 'yahoo', 'ag_news', 'amazon'],
    "6-3": ['dbpedia', 'yahoo', 'ag_news'],
    "6-2": ['dbpedia', 'yahoo'],
    "6-1": ['dbpedia'],
    "7": ['yelp', 'ag_news', 'dbpedia', 'amazon', 'yahoo'],
    "7-4": ['yelp', 'ag_news', 'dbpedia', 'amazon'],
    "7-3": ['yelp', 'ag_news', 'dbpedia'],
    "7-2": ['yelp', 'ag_news'],
    "7-1": ['yelp'],
    "0": ['amazon', 'ag_news'],  # used for debug
    "9": ['amazon'],
    "10": ['ag_news'],
    "11": ['yahoo'],
    "12": ['yelp'],
    "13": ['dbpedia'],
}


@dataclass
class DataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    task: str = "dbpedia"
    num_labels: int = 1

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = dict()

        input_ids, attention_mask, token_type_ids = tuple([torch.tensor(feature[key]) for feature in features] for key in ['input_ids', 'attention_mask', 'token_type_ids'])
        input_ids = nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        token_type_ids = nn.utils.rnn.pad_sequence(
            token_type_ids, batch_first=True, padding_value=0
        )
        label_key = 'labels' if 'labels' in features else 'label'
        labels = [feature[label_key] for feature in features]
        labels = torch.tensor(labels)
        batch.update({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        })
        
        return batch
    

@dataclass
class PreprocessFunction:

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, examples):
        return self.tokenizer(examples['raw_text'], truncation=True, max_length=450)


class GridMaskTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        else:
            self.accelerator.backward(loss)
        
        unwrap_model(model).eliminate_grad_except_task(self.task)

        return loss.detach() / self.args.gradient_accumulation_steps


def train_task(
    task, 
    k_samples, 
    model, 
    tokenizer,
    epoch, 
    tqdm=False,
    lr=1e-3, 
    test=False, 
    batch_size=8, 
    output_dir=None):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    data_name = TASK_TO_DATA_NAME[task]
    accuracy = evaluate.load("accuracy")
    dataset = get_dataset_by_name(data_name, k_samples=k_samples)
    tokenized_dataset = dataset.map(PreprocessFunction(tokenizer=tokenizer), batched=False, num_proc=num_workers)
    tokenized_dataset.set_format(columns=['input_ids', 'token_type_ids', 'attention_mask', 'label', 'raw_text'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, task=task, num_labels=TASK_TO_NUM_LABELS[task])

    training_args = TrainingArguments(
        output_dir=output_dir,
        optim='adamw_torch',
        learning_rate=lr,
        report_to=None,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch,
        weight_decay=0.01,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=0.1,
        save_steps=0.2,
        warmup_steps=100,
        logging_steps=100,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        save_total_limit=1,
        disable_tqdm=not tqdm
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if not test:
        trainer.train()
    else:
        print("================================== Task: {} ==============================".format(task))
        print(trainer.predict(tokenized_dataset["test"]).metrics)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', default="4", type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--output_dir', default="./outputs/all", type=str)
    parser.add_argument('--k_samples', default=-1, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--tqdm', default=False, action='store_true')
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--rk', default=4, type=int)
    args = parser.parse_args()
    if not args.tqdm:
        disable_progress_bar()
    
    num_workers = args.num_workers
    output_dir = args.output_dir
    
    # if args.model_path is None:
    #     config = MultiHeadConfig()
    #     if args.task_id is not None:
    #         config.task_list = TASK_LIST[args.task_id]
    #     model = ScalableLM(config)
    # else:
    #     print("Load checkpoint from: {}".format(args.model_path))
    #     model = ScalableLM.from_pretrained(args.model_path)
    #     config = model.config
    #     if args.task_id is not None:
    #         config.task_list = TASK_LIST[args.task_id]
    config = MultiHeadConfig()
    if args.task_id is not None:
        config.task_list = TASK_LIST[args.task_id]
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    for task in TASK_LIST[args.task_id]:
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=config.task_to_num_label[task])
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        # for name, param in model.named_parameters():
        #     if "LayerNorm" in name:
        #         param.requires_grad = True            
        model.dw = nn.Embedding(768*args.rk, 24)
        for i in range(12):
            nn.init.normal_(model.dw.weight[:, i*2])
            nn.init.constant_(model.dw.weight[:, i*2+1], 0)
            a = model.dw.weight[:, i*2].view(768,args.rk)
            b = model.dw.weight[:, i*2+1].view(args.rk,768)
            model.bert.encoder.layer[i].attention.output.dense.weight = nn.parameter.Parameter(model.bert.encoder.layer[i].attention.output.dense.weight + a@b)
            # model.bert.encoder.layer[i].attention.self.query.weight = nn.parameter.Parameter(model.bert.encoder.layer[i].attention.self.query.weight + a[0]@b[0])
            # model.bert.encoder.layer[i].attention.self.key.weight = nn.parameter.Parameter(model.bert.encoder.layer[i].attention.self.key.weight + a[1]@b[1])
            # model.bert.encoder.layer[i].attention.self.value.weight = nn.parameter.Parameter(model.bert.encoder.layer[i].attention.self.value.weight + a[2]@b[2])
            # for param in  model.bert.encoder.layer[i].attention.output.parameters():
            #     param.requires_grad = True
        train_task(task=task, 
                   k_samples=args.k_samples, 
                   model=model, 
                   tokenizer=tokenizer,
                   epoch=args.epoch, 
                   lr=args.lr, 
                   tqdm=args.tqdm,
                   test=args.test, 
                   batch_size=args.batch_size,
                   output_dir=os.path.join(output_dir, task))
    
        