from PIL import Image
import numpy as np

if __name__=="__main__":
    path = "C:\\CR_\\data\\Train\\2_05c4e98642874de3a1be80218436bc8c.png"
    img = Image.open(path).convert('L')
    long, width = img.size[0], img.size[1]
    img = np.asarray(img)

    # 设置阈值
    threshold_value = 115
    img_new = np.zeros((long, width))

    for i in range(long):
        for j in range(width):
            if img[j][i] > threshold_value:
                img_new[i][j] = 1

    image = Image.fromarray(np.uint8(img_new))
    image.show()

"""
error
"""
