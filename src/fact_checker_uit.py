import time
from typing import List
import re


from bs4 import BeautifulSoup
import requests
import numpy as np
import nltk
from newspaper import Article

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification




def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F300-\U0001FAD6"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def preprocess(texts):
    texts = [text.replace("_", " ") for text in texts]
    texts = [i.lower() for i in texts]
    texts = [remove_emoji(i) for i in texts]

    texts = [re.sub('[^\w\d\s]', '', i) for i in texts]

    texts = [re.sub('\s+|\n', ' ', i) for i in texts]
    texts = [re.sub('^\s|\s$', '', i) for i in texts]

    # texts = [ViTokenizer.tokenize(i) for i in texts]

    return texts


class FactChecker:
    def __init__(self):
        self.load_model()
        self.sess = requests.Session()

    def searchBing(self, question):
        try:
            query = '+'.join(question.split(" "))
            get_header = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }

            url = 'https://www.bing.com/search?q=' + query + "&go=Search&qs=ds&form=QBRE"
            page = self.sess.get(url, headers=get_header)
            soup = BeautifulSoup(page.text, "html.parser")

            output = []

            for searchWrapper in soup.find_all('li', attrs={'class': 'b_algo'}):

                urls = searchWrapper.find_all('a')

                url = [u.get('href') for u in urls if u.get('href')][0]

                # Get True URL from redirect page
                page = self.sess.get(url, headers=get_header)
                soup = BeautifulSoup(page.text, "html.parser")

                script_str = str(soup.find_all('script')[0])
                script_str = [s.strip() for s in script_str.split('\r\n')]
                url = [s for s in script_str if s[:5] == 'var u'][0][9:][:-2]

                title = searchWrapper.find('a').text.strip()

                snippet = searchWrapper.find('p')
                snippet = "" if snippet == None else snippet.text

                # send a GET request to the article URL and retrieve the HTML content
                # article_response = requests.get(url)
                # article_html_content = article_response.content

                # parse the HTML content using BeautifulSoup
                # article_soup = BeautifulSoup(article_html_content, 'html.parser')
                article = Article(url, language='vi')
                try:
                    article.download()
                    article.parse()
                    if len(article.text) > 100:
                        res = {
                                'title': title, 'url': url, 'snippet': snippet,
                                'title_in': article.title, "publish_date": article.publish_date.strftime("%m/%d/%Y, %H:%M:%S") if article.publish_date else article.publish_date,
                                "content": article.text, "keywords":article.keywords,
                                "summary":  article.summary, "author":article.authors }
                        output.append(res)
                except Exception as e:
                    print("exception", e)
            return output

        except Exception as e:
            print(e)

    def query_data(self, x):
        try:
            num_words = 50
            ls_word = x.split(" ")
            message_short = " ".join(ls_word[:num_words])
            rst = self.searchBing(message_short)
            return message_short, rst
        except Exception as e:
            print(e)

    def load_model(self):
        self.device = torch.device("cpu")
        model_name = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_nli = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model_sbert = SentenceTransformer('keepitreal/vietnamese-sbert')

    def get_similarity(self, src_sents, dst_sents, threshold = 0.4):
        
        src_embeddings = self.model_sbert.encode(src_sents)
        dst_embeddings = self.model_sbert.encode(dst_sents)
        dict_score = {}
        for idx_src, emb1 in enumerate(src_embeddings):
            dict_score[idx_src] = {}
            for idx_dst, emb2 in enumerate(dst_embeddings):
                score = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                if score > threshold:
                    dict_score[idx_src][idx_dst] = score
        top_rst = {}
        for src_idx, s_score in dict_score.items():
            sorted_scores = sorted(s_score.items(), key=lambda x: x[1], reverse=True)
            top_rst[src_idx] = {
                "idx": [index for index, _ in sorted_scores],
                "score": [score for _, score in sorted_scores]}
        return dict_score, top_rst

    def predict_nli(self, premise,hypothesis):
        input = self.tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
        output = self.model_nli (input["input_ids"].to(self.device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["Supports", "Neutral", "Refutes"]
        prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
        max_key = max(prediction, key=lambda k: prediction[k])
        return max_key

    def predict(self, claim: str, context: str, top_k: int = 5) -> List:
        t_0 = time.time()
        # import pdb; pdb.set_trace()
        # step 1: crawl evidences from bing search
        # claim_processed = preprocess([claim])[0]
        # query_message, evidences = self.query_data(claim_processed)

        # # evidences = crawler.get_evidences(claim)
        # # step 2: use emebdding setences to search most related setences
        # if len(evidences) == 0:
        #     return None
        # top_evidence = evidences[0]["content"]
        top_evidence = context
        post_message = nltk.tokenize.sent_tokenize(claim)
        evidences = nltk.tokenize.sent_tokenize(top_evidence)
        _, top_rst = self.get_similarity(post_message, evidences)
        ls_verdict = []
        
        ls_evidence = []
         # step 3: use model NLI to predict pair claim-evidence
        for idx_src,src_text in enumerate(post_message):
            # import pdb; pdb.set_trace()
            for idx, idx_e in enumerate(top_rst[idx_src]["idx"][:top_k]):
                e_text = evidences[idx_e]
                score_sim = top_rst[idx_src]["score"][idx]
                verdict_nli = self.predict_nli(src_text, e_text)
                ls_verdict.append(verdict_nli)
                ls_evidence.append(
                            {
                                "claim_sentence": src_text,
                                "evidence_sentence": e_text,
                                "label": verdict_nli,
                                "score_sim": float(score_sim)
                            })
        if not len(ls_evidence):
            return [], "Neutral"
        # ls_verdict = list(filter(lambda x: x != "Neutral",ls_verdict))
        ls_val, ls_count = np.unique(ls_verdict, return_counts = True)
        ls_val_sorted = list(ls_val[np.argsort(-ls_count)])
        ls_count_sorted = list(map(int,ls_count[np.argsort(-ls_count)]))
        final_verdict = ls_val_sorted[0] if len(ls_val_sorted) else "Neutral"
        total_time = time.time() - t_0
        print("Time predict: {} s".format(total_time))
        return ls_evidence, final_verdict
           
