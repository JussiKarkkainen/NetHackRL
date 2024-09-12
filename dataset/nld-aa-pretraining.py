import nle.dataset as nld
import time
from nle.nethack import tty_render

path_to_nld_aa = "nld-aa-taster/nld-aa-taster/nle_data"

dbfilename = "nld_aa_ttyrecs.db"

if not nld.db.exists(dbfilename):
  nld.db.create(dbfilename)
  nld.add_nledata_directory(path_to_nld_aa, "taster-dataset", dbfilename)

db_conn = nld.db.connect(filename=dbfilename)

dataset = nld.TtyrecDataset(
  "taster-dataset",
  batch_size=1,
  seq_length=32,
  dbfilename=dbfilename,
)

minibatch = next(iter(dataset))

for k, v in minibatch.items():
  print(k, v.shape)

print(tty_render(minibatch["tty_chars"][0, 2], minibatch["tty_colors"][0, 2], minibatch["tty_cursor"][0, 2]))
print(tty_render(minibatch["tty_chars"][0, 3], minibatch["tty_colors"][0, 3], minibatch["tty_cursor"][0, 3]))
print(minibatch["keypresses"][0, 2])
