import time
from Centralized.centralized_IMDB import run_centralized_IMDB

start = time.time()
run_centralized_IMDB(4)
end = time.time()

print(f"Run time is `{end - start}")
