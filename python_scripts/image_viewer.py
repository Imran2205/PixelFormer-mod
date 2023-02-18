from PIL import Image
import numpy as np
import plotly.express as px
import cv2
import png
import io

image = "/Users/imrankabir/Desktop/research/semantic_seg_audio_description/PixelFormer-mod/python_scripts/demo.tiff"


def save_image(arr, filename):
    # is_success, buffer = cv2.imencode(".tiff", arr)
    # io_buf = io.BytesIO(buffer)
    # print(type(io_buf))
    # with open(filename, 'wb') as f:
    #     f.write(io_buf.read())
    cv2.imwrite(filename, arr)


def raw_depth_to_gray_32bit(raw_depth, verbose=False):
    depth = raw_depth.convert('RGB')
    depth = np.array(depth)

    if verbose:
        print(np.max(depth), np.min(depth), depth.shape, depth.dtype)

    depth = np.dot(depth[..., :3], [1, 256, 256 * 256]).astype(np.int32)

    if verbose:
        print(np.max(depth), np.min(depth), depth.shape, depth.dtype)

    return depth


def raw_image(image_, verbose=False):
    depth = np.array(image_)

    if verbose:
        print(np.max(depth), np.min(depth), depth.shape, depth.dtype)

    return depth


img = Image.open(image)

# img = raw_depth_to_gray_32bit(img, verbose=True)
img = raw_image(img, verbose=True)

save_image(img, filename='demo.tiff')

fig = px.imshow(img)
fig.show()
