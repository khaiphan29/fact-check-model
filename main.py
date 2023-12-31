#uvicorn main:app --reload 
from fastapi import FastAPI, status
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

from typing import List

import os
import json
import time

from src.myNLI import FactChecker
from src.crawler import MyCrawler

#request body
class Claim(BaseModel):
    claim: str

class ScrapeBase(BaseModel):
    id: int
    name: str
    scraping_url: str

class ScrapeList(BaseModel):
    data: List[ScrapeBase]

app = FastAPI()

# load model
t_0 = time.time()
fact_checker = FactChecker()
t_load = time.time() - t_0
print("time load model: {}".format(t_load))

crawler = MyCrawler()

label_code = {
    "REFUTED": 0,
    "SUPPORTED": 1,
    "NEI": 2
}

@app.get("/")
async def root():
    return {"msg": "This is for interacting with Fact-checking AI Model"}

@app.post("/ai-fact-check")
async def get_claim(req: Claim):
    claim = req.claim
    result = fact_checker.predict(claim)
    print(result)

    if not result:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return { "claim": claim,
            "final_label": label_code[result["label"]],
            "evidence": result["evidence"],
            "provider": result["provider"],
            "url": result["url"]
        }

@app.post("/scraping-check")
async def get_claim(req: ScrapeList):
    response = []
    for ele in req.data:
        response.append({
            "id": ele.id,
            "name": ele.name,
            "scraping_url": ele.scraping_url,
            "status": crawler.scraping(ele.scraping_url)
        })
        

    return JSONResponse({
        "list": response
    })