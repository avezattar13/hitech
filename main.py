from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import pyarrow.dataset as ds
import pyarrow.compute as pc
from huggingface_hub import HfFileSystem
import os
import time
import json

app = FastAPI(title="Hitek")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
REPO_ID = "avezattar/Hitek" 
HF_TOKEN = os.getenv("hf_token") 

global_fs = HfFileSystem(token=HF_TOKEN)

# --- SPEED UPGRADE: Pre-define the exact 10 file paths ---
# This skips the slow glob/listing operation on Hugging Face
FILE_LIST = [f"datasets/{REPO_ID}/final_master_shard_{i}.parquet" for i in range(10)]

@app.get("/")
def root():
    return {"status": "online", "message": "Hitek API is Live."}

@app.get("/search")
def search(mobile: str = Query(..., description="10-digit mobile number")):
    start_time = time.time()
    
    try:
        if len(mobile) != 10 or not mobile.isdigit():
            return Response(
                content=json.dumps({"error": "Please provide a valid 10-digit mobile number."}, indent=4), 
                media_type="application/json"
            )

        # Load the dataset using the pre-defined list
        dataset = ds.dataset(FILE_LIST, filesystem=global_fs, format="parquet")
        
        # Filter for the exact mobile number
        filter_expr = pc.equal(pc.field("mobile"), mobile)
        
        # use_threads=False prevents Vercel 1GB RAM OOM crash
        table = dataset.scanner(filter=filter_expr, use_threads=False).head(5) 
        raw_records = table.to_pylist()
        
        elapsed = time.time() - start_time
        
        response_data = {
            "meta": {
                "credits": "Avez",
                "search_query": mobile,
                "total_records": len(raw_records),
                "execution_time": f"{elapsed:.3f}s"
            },
            "results": []
        }

        if not raw_records:
            response_data["message"] = "No record found in the database."
            return Response(
                content=json.dumps(response_data, indent=4), 
                media_type="application/json"
            )
            
        for i, row in enumerate(raw_records, 1):
            clean_row = {}
            if row.get("mobile"): clean_row["mobile"] = row["mobile"]
            if row.get("name"): clean_row["name"] = row["name"]
            if row.get("fname"): clean_row["father_name"] = row["fname"]
            if row.get("address"): clean_row["address"] = row["address"]
            if row.get("alt"): clean_row["alternate_mobile"] = row["alt"]
            if row.get("circle"): clean_row["telecom_circle"] = row["circle"]
            if row.get("id"): clean_row["record_id"] = row["id"]
            if row.get("email"): clean_row["email"] = row["email"]
            
            clean_row["result_id"] = i
            response_data["results"].append(clean_row)
            
        return Response(
            content=json.dumps(response_data, indent=4, default=str), 
            media_type="application/json"
        )
        
    except Exception as e:
        return Response(
            content=json.dumps({"error": str(e)}, indent=4), 
            media_type="application/json"
        )
