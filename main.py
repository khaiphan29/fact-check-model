#uvicorn main:app --reload 
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import os
import json
import time

from src.myNLI import FactChecker
#request body
class Claim(BaseModel):
    claim: str

app = FastAPI()

# load model
t_0 = time.time()
fact_checker = FactChecker()
t_load = time.time() - t_0
print("time load model: {}".format(t_load))

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
        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content={})

    return { "claim": claim,
            "final_label": label_code[result["label"]],
            "evidence": result["evidence"],
            "provider": result["provider"],
            "url": result["url"]
        }