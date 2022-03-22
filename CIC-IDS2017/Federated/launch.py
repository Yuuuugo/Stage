import threading
import os
import grpc
from Data import nb_client

if os.environ.get('https_proxy'):
 del os.environ['https_proxy']
if os.environ.get('http_proxy'):
 del os.environ['http_proxy']

semaphore = threading.Semaphore(nb_client)

def run_command(cmd):
    with semaphore:
        os.system(cmd)

for i in range(nb_client):
    threading.Thread(target=run_command, args=("python3 client.py --client=" +str(i), )).start()