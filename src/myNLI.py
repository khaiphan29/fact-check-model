import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import nltk

# import datasets
from datasets import Dataset, DatasetDict

from typing import List

from .utils import timer_func
from .nli_v3 import NLI_model
from .crawler import MyCrawler

int2label = {0:'SUPPORTED', 1:'NEI', 2:'REFUTED'}

class FactChecker:

    @timer_func
    def __init__(self):
        self.INPUT_TYPE = "mean"
        self.load_model()

    @timer_func
    def load_model(self):
        self.envir = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")    # LOAD mDEBERTa TOKENIZER
        self.mDeBertaModel = AutoModel.from_pretrained(f"src/mDeBERTa (ft) V6/mDeBERTa-v3-base-mnli-xnli-{self.INPUT_TYPE}")  # LOAD FINETUNED MODEL
        # Load classifier model
        self.checkpoints = torch.load(f"src/mDeBERTa (ft) V6/{self.INPUT_TYPE}.pt", map_location=self.envir)

        self.classifierModel = NLI_model(768, torch.tensor([0., 0., 0.])).to(self.envir)
        self.classifierModel.load_state_dict(self.checkpoints['model_state_dict'])

        #Load model for predict similarity
        self.model_sbert = SentenceTransformer('keepitreal/vietnamese-sbert')
    
    @timer_func
    def get_similarity_v2(self, src_sents, dst_sents, threshold = 0.4):
        corpus_embeddings = self.model_sbert.encode(dst_sents, convert_to_tensor=True)
        top_k = min(5, len(dst_sents))
        ls_top_results = []
        for query in src_sents:
            query_embedding = self.model_sbert.encode(query, convert_to_tensor=True)
            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            # print("\n\n======================\n\n")
            # print("Query:", src_sents)
            # print("\nTop 5 most similar sentences in corpus:")
            ls_top_results.append({
                "top_k": top_k,
                "claim": query,
                "sim_score": top_results,
                "evidences": [dst_sents[idx] for _, idx in zip(top_results[0], top_results[1])],
            })

            # for score, idx in zip(top_results[0], top_results[1]):
            #     print(dst_sents[idx], "(Score: {:.4f})".format(score))
        return None,ls_top_results
    
    @timer_func
    def inferSample(self, evidence, claim):

        @timer_func
        def mDeBERTa_tokenize(data): # mDeBERTa model: Taking input_ids
            premises = [premise for premise, _ in data['sample']]
            hypothesis = [hypothesis for _, hypothesis in data['sample']]

            with torch.no_grad():
                input_token = (self.tokenizer(premises, hypothesis, truncation=True, return_tensors="pt", padding = True)['input_ids']).to(self.envir)
                embedding = self.mDeBertaModel(input_token).last_hidden_state

            mean_embedding = torch.mean(embedding[:, 1:, :], dim = 1)
            cls_embedding = embedding[:, 0, :]

            return {'mean':mean_embedding, 'cls':cls_embedding}

        @timer_func
        def predict_mapping(batch):
            with torch.no_grad():
                predict_label, predict_prob = self.classifierModel.predict_step((batch[self.INPUT_TYPE].to(self.envir), None))
            return {'label':predict_label, 'prob':-predict_prob}

        # Mapping the predict label into corresponding string labels
        @timer_func
        def output_predictedDataset(predict_dataset):
            for record in predict_dataset:
                labels = int2label[ record['label'].item() ]
                confidence = record['prob'].item()

            return {'labels':labels, 'confidence':confidence}

        dataset = {'sample':[(evidence, claim)], 'key': [0]}
        output_dataset = DatasetDict({
            'infer': Dataset.from_dict(dataset)
        })

        @timer_func
        def tokenize_dataset():

            tokenized_dataset = output_dataset.map(mDeBERTa_tokenize, batched=True, batch_size=1)
            return tokenized_dataset

        tokenized_dataset = tokenize_dataset()
        tokenized_dataset = tokenized_dataset.with_format("torch", [self.INPUT_TYPE, 'key'])
        # Running inference step
        predicted_dataset = tokenized_dataset.map(predict_mapping, batched=True, batch_size=tokenized_dataset['infer'].num_rows)
        return output_predictedDataset(predicted_dataset['infer'])
    
    @timer_func
    def predict_vt(self, claim: str) -> List:
        # import pdb; pdb.set_trace()
        # step 1: crawl evidences from bing search
        crawler = MyCrawler()
        evidences = crawler.searchGoogle(claim)

        # evidences = crawler.get_evidences(claim)
        # step 2: use emebdding setences to search most related setences
        if len(evidences) == 0:
            return None
        
        for evidence in evidences:
            print(evidence['url'])
            top_evidence = evidence["content"]

            post_message = nltk.tokenize.sent_tokenize(claim)
            evidences = nltk.tokenize.sent_tokenize(top_evidence)
            _, top_rst = self.get_similarity_v2(post_message, evidences)

            print(top_rst)

            ls_evidence, final_verdict = self.get_result_nli_v2(top_rst)

            print("FINAL: " + final_verdict)
        # _, top_rst = self.get_similarity_v1(post_message, evidences)
        # ls_evidence, final_verdict = self.get_result_nli_v1(post_message, top_rst, evidences)
        return ls_evidence, final_verdict
           

    @timer_func
    def predict(self, claim):
        crawler = MyCrawler()
        evidences = crawler.searchGoogle(claim)

        if evidences:
            tokenized_claim = nltk.tokenize.sent_tokenize(claim)
            evidence = evidences[0]
            tokenized_evidence = nltk.tokenize.sent_tokenize(evidence["content"])
            # print("TOKENIZED EVIDENCES")
            # print(tokenized_evidence)
            _, top_rst = self.get_similarity_v2(tokenized_claim, tokenized_evidence)
            
            processed_evidence = "\n".join(top_rst[0]["evidences"])
            print(processed_evidence)

            nli_result = self.inferSample(processed_evidence, claim)
            return {
                "claim": claim,
                "label": nli_result["labels"],
                "confidence": nli_result['confidence'],
                "evidence": processed_evidence if nli_result["labels"] != "NEI" else "",
                "provider": evidence['provider'],
                "url": evidence['url']
            }
        
            

    @timer_func
    def predict_nofilter(self, claim):
        crawler = MyCrawler()
        evidences = crawler.searchGoogle(claim)
        tokenized_claim = nltk.tokenize.sent_tokenize(claim)

        evidence = evidences[0]

        processed_evidence = evidence['content']

        nli_result = self.inferSample(processed_evidence, claim)
        return {
            "claim": claim,
            "label": nli_result["labels"],
            "confidence": nli_result['confidence'],
            "evidence": processed_evidence if nli_result["labels"] != "NEI" else "",
            "provider": evidence['provider'],
            "url": evidence['url']
        }