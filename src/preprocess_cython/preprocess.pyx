import numpy as np
cimport numpy as np

def preprocess_char_image_cython(np.ndarray[np.uint8_t, ndim=2] chars, 
                              np.ndarray[np.int8_t, ndim=2] colors, 
                              np.ndarray[np.uint8_t, ndim=5] cache_array):

  cdef int i, j, color
  cdef np.ndarray[np.uint8_t, ndim=3] cha
  cdef np.ndarray[np.float32_t, ndim=3] pixel_obs = np.zeros((9*12, 9*12, 3), dtype=np.float32)
  
  for i in range(12):
      for j in range(12):
          color = colors[i, j]
          cha = cache_array[chars[i, j], color]
          pixel_obs[i*9:(i+1)*9, j*9:(j+1)*9, :] = cha
  return pixel_obs
