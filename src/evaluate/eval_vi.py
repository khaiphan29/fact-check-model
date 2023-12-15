import pandas as pd
import time
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=['Refutes', 'Supports', 'Neutral'])

    # Create a heatmap of the confusion matrix
    fig, ax = plt.subplots()
    heatmap = sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', xticklabels=['Refutes', 'Supports', 'Neutral'], yticklabels=['Refutes', 'Supports', 'Neutral'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # Save the heatmap as an image
    heatmap.figure.savefig(PATH_SAVE + "/"+"confusion_matrix.png")
    

class ModelNLI:
    def __init__(self):
        self.device = torch.device("cpu")
        model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_nli = AutoModelForSequenceClassification.from_pretrained(model_name)
    def predict(self, premise,hypothesis):
        input = self.tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
        output = self.model_nli (input["input_ids"].to(self.device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["Supports", "Neutral", "Refutes"]
        prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
        max_key = max(prediction, key=lambda k: prediction[k])
        return max_key
PATH_DATA = "/Users/tham.tran2/Documents/Master/data/health-ver/translate"
TEST_FILE = "healthver_test_vi.xlsx"
DEV_FILE = "healthver_dev_vi.xlsx"
TRAIN_FILE = "healthver_train_vi.xlsx"
PATH_SAVE = "tmp/model_rst"
def main():
    t_0 = time.time()
    df = pd.read_excel(PATH_DATA+"/"+TEST_FILE)
    print(df.shape)
    model = ModelNLI()
    df["label"] = df[" nhãn"].replace([' Trung lập', ' Hỗ trợ', ' bác bỏ'], ["Neutral","Supports", "Refutes"])
    df["predicted"] = df.apply(lambda x: model.predict(x[" chứng cớ"], x[" khẳng định"]), axis = 1)
    df.to_csv(PATH_SAVE + "/"+"result.csv", index = False)
    evaluate(df["label"],  df["predicted"])
    print("TOTAL TIME: {}".format(time.time() - t_0))
if __name__ == "__main__":
    main()