from bs4 import BeautifulSoup
import requests


headers = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36",
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

page = requests.get("https://www.bing.com/search?", headers=headers, params={"q": "Filip Nguyễn đủ điều kiện dự Asian Cup 2024"})

print(page.url)
soup = BeautifulSoup(page.text, 'html.parser')

anchors = soup.find_all("a")
for anchor in anchors:
    if anchor is not None:
        try:
            if "http" in anchor["href"]:
                print(anchor["href"])
        except KeyError:
            continue