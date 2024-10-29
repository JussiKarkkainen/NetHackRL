import numpy as np
cimport numpy as np

def preprocess_char_image_cython(np.ndarray[np.uint8_t, ndim=3] out_image,
                                 np.ndarray[np.uint8_t, ndim=2] chars,
                                 np.ndarray[np.int8_t, ndim=2] colors,
                                 int out_width_char, int out_height_char,
                                 int offset_h, int offset_w,
                                 np.ndarray[np.uint8_t, ndim=5] cache_array):

  cdef int h, w, h_char, w_char, char, color, h_pixel, w_pixel

  for h in range(out_height_char):
    h_char = h + offset_h
    if h_char < 0 or h_char >= chars.shape[0]:
      continue
    for w in range(out_width_char):
      w_char = w + offset_w
      if w_char < 0 or w_char >= chars.shape[1]:
        continue
      
      char = chars[h_char, w_char]
      color = colors[h_char, w_char]

      h_pixel = h * 9
      w_pixel = w * 9

      out_image[h_pixel : h_pixel + 9, w_pixel : w_pixel + 9, :] = cache_array[char, color]

  return out_image
