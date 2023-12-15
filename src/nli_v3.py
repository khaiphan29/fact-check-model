import torch
from torch import nn as nn
import pandas as pd

from transformers import AutoModel, AutoTokenizer

# import datasets
from datasets import Dataset, DatasetDict

from sklearn.metrics import classification_report
from sklearn.metrics._classification import _check_targets

envir = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

int2label = {0:'SUPPORTED', 1:'NEI', 2:'REFUTED'}

class NLI_model(nn.Module):
    def __init__(self, input_dims, class_weights=torch.tensor([0., 0., 0.])):
        super(NLI_model, self).__init__()

        self.classification = nn.Sequential(
            nn.Linear(input_dims, 3)
        )

        self.criterion = nn.CrossEntropyLoss(class_weights)

    def forward(self, input):
        output_linear = self.classification(input)
        return output_linear

    def training_step(self, train_batch, batch_idx=0):
        input_data, targets = train_batch
        outputs = self.forward(input_data)
        loss = self.criterion(outputs, targets)
        return loss

    def predict_step(self, batch, batch_idx=0):
        input_data, _ = batch
        outputs = self.forward(input_data)
        prob = outputs.softmax(dim = -1)
        sort_prob, sort_indices = torch.sort(-prob, 1)
        return sort_indices[:,0], sort_prob[:,0]

    def validation_step(self, val_batch, batch_idx=0):
        _, targets = val_batch
        sort_indices, _ = self.predict_step(val_batch, batch_idx)
        report = classification_report(list(targets.to('cpu').numpy()), list(sort_indices.to('cpu').numpy()), output_dict=True, zero_division = 1)
        return report

    def test_step(self, batch, dict_form, batch_idx=0):
        _, targets = batch
        sort_indices, _ = self.predict_step(batch, batch_idx)
        report = classification_report(targets.to('cpu').numpy(), sort_indices.to('cpu').numpy(), output_dict=dict_form, zero_division = 1)
        return report

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 1e-5)


def inferSample(evidence, claim, tokenizer, mDeBertaModel, classifierModel, input_type):

    def mDeBERTa_tokenize(data): # mDeBERTa model: Taking input_ids
        premises = [premise for premise, _ in data['sample']]
        hypothesis = [hypothesis for _, hypothesis in data['sample']]

        with torch.no_grad():
            input_token = (tokenizer(premises, hypothesis, truncation=True, return_tensors="pt", padding = True)['input_ids']).to(envir)
            embedding = mDeBertaModel(input_token).last_hidden_state

        mean_embedding = torch.mean(embedding[:, 1:, :], dim = 1)
        cls_embedding = embedding[:, 0, :]

        return {'mean':mean_embedding, 'cls':cls_embedding}

    def predict_mapping(batch):
        with torch.no_grad():
            predict_label, predict_prob = classifierModel.predict_step((batch[input_type].to(envir), None))
        return {'label':predict_label, 'prob':-predict_prob}

    # Mapping the predict label into corresponding string labels
    def output_predictedDataset(predict_dataset):
        for record in predict_dataset:
            labels = int2label[ record['label'].item() ]
            confidence = record['prob'].item()

        return {'labels':labels, 'confidence':confidence}

    dataset = {'sample':[(evidence, claim)], 'key': [0]}

    output_dataset = DatasetDict({
        'infer': Dataset.from_dict(dataset)
    })

    tokenized_dataset = output_dataset.map(mDeBERTa_tokenize, batched=True, batch_size=1)
    tokenized_dataset = tokenized_dataset.with_format("torch", [input_type, 'key'])

    # Running inference step
    predicted_dataset = tokenized_dataset.map(predict_mapping, batched=True, batch_size=tokenized_dataset['infer'].num_rows)
    return output_predictedDataset(predicted_dataset['infer'])

if __name__ == '__main__':
    # CHANGE 'INPUT_TYPE' TO CHANGE MODEL
    INPUT_TYPE = 'mean' # USE "MEAN" OR "CLS" LAST HIDDEN STATE
    
    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")    # LOAD mDEBERTa TOKENIZER
    mDeBertaModel = AutoModel.from_pretrained(f"src/mDeBERTa (ft) V6/mDeBERTa-v3-base-mnli-xnli-{INPUT_TYPE}")  # LOAD FINETUNED MODEL
    # Load classifier model
    checkpoints = torch.load(f"src/mDeBERTa (ft) V6/{INPUT_TYPE}.pt", map_location=envir)
    classifierModel = NLI_model(768, torch.tensor([0., 0., 0.])).to(envir)
    classifierModel.load_state_dict(checkpoints['model_state_dict'])
    
    evidence = "Sau khi thẩm định, Liên đoàn Bóng đá châu Á AFC xác nhận thủ thành mới nhập quốc tịch của Việt Nam Filip Nguyễn đủ điều kiện thi đấu ở Asian Cup 2024."
    claim = "Filip Nguyễn đủ điều kiện dự Asian Cup 2024"
    print(inferSample(evidence, claim, tokenizer, mDeBertaModel, classifierModel, INPUT_TYPE))