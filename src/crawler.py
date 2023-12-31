import requests
from bs4 import BeautifulSoup
import re
import time

from .utils import timer_func

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


class MyCrawler:
    headers = {
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36",
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    # headers = {
    #             'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:49.0) Gecko/20100101 Firefox/49.0',
    #             # 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    #             # 'Accept-Language': 'en-US,en;q=0.5',
    #             # 'Accept-Encoding': 'gzip, deflate',
    #             # 'DNT': '1',
    #             # 'Connection': 'keep-alive',
    #             # 'Upgrade-Insecure-Requests': '1'
    #         }

    def getSoup(self, url: str):
        req = requests.get(url,headers=self.headers)
        return BeautifulSoup(req.text, 'html.parser')

    def crawl_byContainer(self, url: str, article_container: str, body_class: str):
        soup = self.getSoup(url)

        paragraphs = soup.find(article_container,{"class": body_class})
        if paragraphs:
            #Crawl all paragraphs
            contents = []
            numOfParagraphs = 0
            for p in paragraphs.find_all("p"):
                contents.append(p.get_text())
                numOfParagraphs += 1
                # if numOfParagraphs > 10:
                #     break

            if contents:
                result = "\n".join(contents)
                if (url.split("/")[2] == "vnexpress.net"):
                    result = self.crawl_byElement(soup, "p", "description") + "\n" + result

                return result
        return ""
    
    def crawl_byElement(self, soup, element: str, ele_class: str):
        print("by Elements...")

        paragraph = soup.find(element,{"class": ele_class})
        if paragraph:
            print(paragraph.get_text())
            return paragraph.get_text()
        return ""

    def crawl_webcontent(self, url: str):

        provider = url.split("/")[2]
        content = ""

        if provider == "thanhnien.vn" or provider == "tuoitre.vn":
            content = self.crawl_byContainer(url, "div", "afcbc-body")
        elif provider == "vietnamnet.vn":
            content = self.crawl_byContainer(url, "div", "maincontent")
        elif provider == "vnexpress.net":
            content = self.crawl_byContainer(url, "article", "fck_detail")
        elif provider == "www.24h.com.vn":
            content = self.crawl_byContainer(url, "article", "cate-24h-foot-arti-deta-info")
        elif provider == "vov.vn":
            content = self.crawl_byContainer(url, "div", "article-content")
        elif provider == "vtv.vn":
            content = self.crawl_byContainer(url, "div", "ta-justify")
        elif provider == "vi.wikipedia.org":
            content = self.crawl_byContainer(url, "div", "mw-content-ltr")
        elif provider == "www.vinmec.com":
            content = self.crawl_byContainer(url, "div", "block-content")

        elif provider == "vietstock.vn":
            content = self.crawl_byContainer(url, "div", "single_post_heading")
        elif provider == "vneconomy.vn":
            content = self.crawl_byContainer(url, "article", "detail-wrap")

        elif provider == "dantri.com.vn":
            content = self.crawl_byContainer(url, "article", "singular-container")
            
        # elif provider == "plo.vn":
        #     content = self.crawl_byContainer(url, "div", "article__body")
        
        return provider, url, content
    
    #def crawl_redir(url):

    @timer_func
    def search(self, claim: str, count: int = 1):
        processed_claim = preprocess([claim])[0]

        num_words = 100
        ls_word = processed_claim.split(" ")
        claim_short = " ".join(ls_word[:num_words])

        print(claim_short)
        query = claim_short
        # query = '+'.join(claim_short.split(" "))

        try:

            # print(soup.prettify())

            #get all URLs
            attemp_time = 0
            urls = []
            while len(urls) == 0 and attemp_time < 3:
                req=requests.get("https://www.bing.com/search?", headers=self.headers, params={
                    "q": query, 
                    "responseFilter":"-images",
                    "responseFilter":"-videos"
                    })
                print("Query URL: " + req.url)

                print("Crawling Attempt " + str(attemp_time))
                soup = BeautifulSoup(req.text, 'html.parser')

                completeData = soup.find_all("li",{"class":"b_algo"})
                for data in completeData: 
                    urls.append(data.find("a", href=True)["href"])
                attemp_time += 1
                time.sleep(1)
            
            print("Got " + str(len(urls)) + " urls")

            result = []

            for url in urls:
                print("Crawling... " + url)
                provider, url, content = self.crawl_webcontent(url)

                if content:
                    result.append({
                        "provider": provider,
                        "url": url,
                        "content": content
                    })
                    count -= 1
                    if count == 0:
                        break

            return result

        except Exception as e:
            print(e)
            return []

    @timer_func
    def searchGoogle(self, claim: str, count: int = 1):
        processed_claim = preprocess([claim])[0]

        num_words = 100
        ls_word = processed_claim.split(" ")
        claim_short = " ".join(ls_word[:num_words])

        print(claim_short)
        query = claim_short
        # query = '+'.join(claim_short.split(" "))

        try:

            # print(soup.prettify())

            #get all URLs
            attemp_time = 0
            urls = []
            while len(urls) == 0 and attemp_time < 3:
                req=requests.get("https://www.google.com/search?", headers=self.headers, params={
                    "q": query
                    })
                print("Query URL: " + req.url)

                print("Crawling Attempt " + str(attemp_time))
                soup = BeautifulSoup(req.text, 'html.parser')

                completeData = soup.find_all("a",{"jsname":"UWckNb"})
                for data in completeData: 
                    urls.append(data["href"])
                attemp_time += 1
                time.sleep(1)
            
            print("Got " + str(len(urls)) + " urls")

            result = []

            for url in urls:
                print("Crawling... " + url)
                provider, url, content = self.crawl_webcontent(url)

                if content:
                    result.append({
                        "provider": provider,
                        "url": url,
                        "content": content
                    })
                    count -= 1
                    if count == 0:
                        break

            return result

        except Exception as e:
            print(e)
            return []
        
    @timer_func
    def scraping(self, url: str):
        try:
            provider, url, content = self.crawl_webcontent(url)

            if content:
                return True
            return False

        except Exception as e:
            print(e)
            return False