import time
from typing import List


from bs4 import BeautifulSoup
import requests
import numpy as np
import nltk
from newspaper import Article

import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .crawler import MyCrawler


from .utils import timer_func

class FactChecker:
    @timer_func
    def __init__(self):
        self.load_model()

    @timer_func
    def query_data(self, x):
        try:
            num_words = 50
            ls_word = x.split(" ")
            message_short = " ".join(ls_word[:num_words])
            rst = self.searchBing(message_short)
            return message_short, rst
        except Exception as e:
            print(e)

    @timer_func
    def load_model(self):
        self.device = torch.device("cpu")
        model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_nli = AutoModelForSequenceClassification.from_pretrained(model_name)
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
    def predict_nli(self, premise,hypothesis):
        input = self.tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
        output = self.model_nli (input["input_ids"].to(self.device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["Supports", "Neutral", "Refutes"]
        prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
        max_key = max(prediction, key=lambda k: prediction[k])
        return max_key
    
    @timer_func
    def get_result_nli_v2(self, top_rst):
        ls_verdict = []
        ls_evidence = []
         # step 3: use model NLI to predict pair claim-evidence
        for rst in top_rst:

            claim = rst["claim"]
            for evidence in rst["evidences"]:
                # import pdb; pdb.set_trace()
                verdict_nli = self.predict_nli(rst["claim"], evidence)
                ls_verdict.append(verdict_nli)
                ls_evidence.append(
                            {
                                "claim_sentence": claim,
                                "evidence_sentence": evidence,
                                "label": verdict_nli
                            })
                # print("---")
                # print(evidence)
                print(verdict_nli)
                # print("---")
        if not len(ls_evidence):
            return [], "Neutral"
        final_verdict = ls_evidence[0]["label"]
        # ls_verdict = list(filter(lambda x: x != "Neutral",ls_verdict))
        # ls_verdict = [verdict for verdict in ls_verdict if verdict !=  ]
        # ls_val, ls_count = np.unique(ls_verdict, return_counts = True)
        # ls_val_sorted = list(ls_val[np.argsort(-ls_count)])
        # ls_count_sorted = list(map(int,ls_count[np.argsort(-ls_count)]))
        # final_verdict = ls_val_sorted[0] if len(ls_val_sorted) else "Neutral"
        return ls_evidence, final_verdict
    
    @timer_func
    def predict_vt(self, claim: str) -> List:
        # import pdb; pdb.set_trace()
        # step 1: crawl evidences from bing search
        crawler = MyCrawler()
        evidences = crawler.search(claim)

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

            # print(top_rst)

            ls_evidence, final_verdict = self.get_result_nli_v2(top_rst)

            print("FINAL: " + final_verdict)
        # _, top_rst = self.get_similarity_v1(post_message, evidences)
        # ls_evidence, final_verdict = self.get_result_nli_v1(post_message, top_rst, evidences)
        return ls_evidence, final_verdict
           
