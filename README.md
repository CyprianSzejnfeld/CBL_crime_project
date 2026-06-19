# How to reproduce

Notice that full run from the raw data can take from 20 - 40 minutes.

Download raw data from here https://drive.google.com/drive/folders/1kfYzrauaYv-bxQEf9HM_g1REP4RKvGFq?usp=sharing and in put it in data/raw folder.

If there is no data folder create it in the root folder among folders like backend. Put the downloaded raw folder inside it.


## 1. Python env

Run this commands one by one in the terminal being at the root file level:

python3.13 -m venv backend/.venv
backend/.venv/bin/python -m pip install --upgrade pip
backend/.venv/bin/python -m pip install -r backend/requirements.txt


## 2. Build the data

Run this in the terminal being at the root level:

bash run_pipeline.sh


## 3. Backend (terminal 1)


cd backend
.venv/bin/python -m uvicorn app.main:app --port 8000


Run this:
## 4. Frontend (terminal 2)


cd frontend
npm install
npm run dev


Open http://localhost:5173

