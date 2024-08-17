import os

import requests
from dotenv import load_dotenv

load_dotenv()





result = stt(os.getenv("UZBEKVOICE_API_KEY"), file_path)
print(result)
