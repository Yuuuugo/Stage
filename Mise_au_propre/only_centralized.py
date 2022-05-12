import time
from Centralized.centralized_IMDB import run_centralized_IMDB

start = time.time()
run_centralized_IMDB(20)
end = time.time()

print(f"Run time is `{end - start}")
