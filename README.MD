- add data to data/dumps/wb/wb/... and data/dumps/fedstatru/fedstatru/...
- create virtual environment
- create .env file and fill it like in .env.example
pip install -r requirements.txt
pip install -r .\requirements-vector.txt
python build_indexes.py
docker compose up -d
use at http://localhost:8501