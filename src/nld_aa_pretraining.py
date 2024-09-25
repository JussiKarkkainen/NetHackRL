import nle.dataset as nld
import time
from nle.nethack import tty_render
from nle.env.tasks import NetHackChallenge, NetHackScore
from concurrent.futures import ThreadPoolExecutor
from tinygrad import Tensor, nn

def dataset(batch_size=32, seq_len=256):
  path_to_nld_aa = "../dataset/nld-aa/nle_data"

  dbfilename = "../dataset/nld_aa_actual_ttyrecs.db"

  if not nld.db.exists(dbfilename):
    nld.db.create(dbfilename)
    nld.add_nledata_directory(path_to_nld_aa, "nld_aa_actual", dbfilename)

  db_conn = nld.db.connect(filename=dbfilename)

  subselect_sql = "SELECT gameid FROM games WHERE role=? AND race=?"
  subselect_sql_args = ("Mon", "Hum")

  with ThreadPoolExecutor(max_workers=10) as tp:
    dataset = nld.TtyrecDataset(
      "nld_aa_actual",
      batch_size=batch_size,
      seq_length=seq_len,
      shuffle=True,
      loop_forever=False,
      dbfilename=dbfilename,
      subselect_sql=subselect_sql,
      subselect_sql_args=subselect_sql_args,
      threadpool=tp
    )

    print(f"Human Monk dataset has: {len(dataset._gameids)} games.")

    env = NetHackScore(savedir=None, character="@")
    embed_actions = Tensor.zeros((256, 1))
    for i, a in enumerate(env.actions):
      embed_actions[a.value][0] = i

    embedding_layer = nn.Embedding(256, 1)
    embedding_layer.weight = embed_actions

    for minibatch in dataset:
      keypresses = Tensor(minibatch["keypresses"])
      actions = embedding_layer(keypresses).squeeze(-1)
      m = {k: minibatch[k] for k in ["tty_chars", "tty_colors", "done"]}
      m["actions"] = actions
      yield m

