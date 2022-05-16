import time
from Centralized.centralized_JS import run_centralized_JS

start = time.time()
run_centralized_JS(4, "test")
end = time.time()

print(f"Run time is `{end - start}")
