import nle.dataset as nld
import time
import torch
from nle.nethack import tty_render
from nle.env.tasks import NetHackChallenge, NetHackScore

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

  env = NetHackScore(savedir=None, character="@")
  embed_actions = torch.zeros((256, 1))
  for i, a in enumerate(env.actions):
    embed_actions[a.value][0] = i

  embed_actions = torch.nn.Embedding.from_pretrained(embed_actions)

  for minibatch in dataset:
    keypresses = torch.Tensor(minibatch["keypresses"]).long()
    actions = embed_actions(keypresses).squeeze(-1).long()
    m = {k: minibatch[k] for k in ["tty_chars", "tty_colors", "done"]}
    m["actions"] = actions
    yield m

if __name__ == "__main__":
  dataset()

