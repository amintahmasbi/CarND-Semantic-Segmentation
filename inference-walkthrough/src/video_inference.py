# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import scipy.misc
from PIL import Image
import numpy as np

from graph_utils import load_graph


# The input is an NumPy array.
# The output should also be a NumPy array.
def pipeline(img):
    
    image = Image.fromarray(img)
    image = scipy.misc.imresize(image, image_shape)
    
    im_softmax = sess.run([softmax], {image_input: [image], keep_prob: 1.0})
    
    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    
    return np.asarray(street_im, dtype=np.uint8)



image_shape = (160, 576)
clip = VideoFileClip('driving.mp4')

sess, _ = load_graph('optimized_graph.pb')
graph = sess.graph

image_input = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
softmax = graph.get_tensor_by_name('softmax:0')

    
new_clip = clip.fl_image(pipeline)
    
# write to file
new_clip.write_videofile('result.mp4')


