from tinygrad import Tensor
import multiprocessing as mp

def worker():
  a = Tensor([1, 2, 3, 4])
  b = Tensor([1, 2, 3, 4])
  print(a.add(b).sum().item())

if __name__ == "__main__":
  #mp.set_start_method("spawn")
  #data_process = mp.Process(target=worker)
  #data_process.start()
  worker()
  
