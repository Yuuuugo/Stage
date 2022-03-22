import sys
import tensorflow as tf
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/Users/hugo/Stage/Stage/CIC-IDS2017/Dataset')



nb_client = 5
nb_rounds = 3
seed = tf.random.set_seed(
    42
)




