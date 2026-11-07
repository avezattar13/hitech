from fastapi import FastAPI, Query, Request
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
REPO_ID = "avezattar/Hitek-bucket" 
HF_TOKEN = os.getenv("hf_token") 

# --- API KEY CONFIGURATION ---
# Fetch the keys from Vercel env var, split by comma, and strip whitespace
RAW_KEYS = os.getenv("api_keys", "")
VALID_API_KEYS = [k.strip() for k in RAW_KEYS.split(",") if k.strip()]

global_fs = HfFileSystem(token=HF_TOKEN)

FILE_LIST = [f"buckets/{REPO_ID}/final_master_shard_{i}.parquet" for i in range(10)]

@app.get("/")
def root():
    return {"status": "online", "message": "Hitek API is Live."}

# The path expects: /api_key=YOUR_KEY&search
# We capture the key using a path parameter before the &search part
@app.get("/api_key={api_key}&search")
def search(
    request: Request, 
    api_key: str, 
    mobile: str = Query(..., description="10-digit mobile number")
):
    start_time = time.time()
    
    # 1. Validate API Key
    if not VALID_API_KEYS or api_key not in VALID_API_KEYS:
        return Response(
            content=json.dumps({"error": "Unauthorized. Invalid or missing API key."}, indent=4), 
            media_type="application/json",
            status_code=401
        )

    try:
        if len(mobile) != 10 or not mobile.isdigit():
            return Response(
                content=json.dumps({"error": "Please provide a valid 10-digit mobile number."}, indent=4), 
                media_type="application/json",
                status_code=400
            )

        dataset = ds.dataset(FILE_LIST, filesystem=global_fs, format="parquet")
        filter_expr = pc.equal(pc.field("mobile"), mobile)
        
        # Pruning columns saves memory, batch size prevents OOM
        columns_to_read = ["mobile", "name", "fname", "address", "alt", "circle", "id", "email"]
        
        scanner = dataset.scanner(
            filter=filter_expr,
            columns=columns_to_read,
            use_threads=False,
            batch_size=1024 
        )
        
        raw_records = []
        for batch in scanner.to_batches():
            if batch.num_rows > 0:
                raw_records = batch.to_pylist()
                break
                
        elapsed = time.time() - start_time
        
        response_data = {
            "meta": {
                "credits": "Avezzz",
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
            media_type="application/json",
            status_code=500
        )
