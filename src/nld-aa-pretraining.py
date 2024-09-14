import nle.dataset as nld
import time
from nle.nethack import tty_render

def dataset(batch_size=32, seq_len=256):
  path_to_nld_aa = "../dataset/nld-aa-taster/nld-aa-taster/nle_data"

  dbfilename = "../dataset/nld_aa_ttyrecs.db"

  if not nld.db.exists(dbfilename):
    nld.db.create(dbfilename)
    nld.add_nledata_directory(path_to_nld_aa, "taster-dataset", dbfilename)

  db_conn = nld.db.connect(filename=dbfilename)

  dataset = nld.TtyrecDataset(
    "taster-dataset",
    batch_size=batch_size,
    seq_length=seq_len,
    dbfilename=dbfilename,
  )

  for minibatch in dataset:
    m = {k: minibatch[k] for k in ["tty_chars", "tty_colors"]}
    yield m


if __name__ == "__main__":
  dataset_gen = dataset()

