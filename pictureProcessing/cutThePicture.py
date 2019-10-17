from PIL import Image
import numpy as np

if __name__=="__main__":
    path = "C:\\CR_\\data\\1_0f1c21350599491c89ebc8eef45e272e.jpg"
    img = Image.open(path).convert('RGB')
    wide, high = img.size[0], img.size[1]  # 2976*3968
    img = np.asarray(img)

    # 设置窗口大小
    windows_size = 500

    wid_num = int(wide/windows_size)
    hig_num = int(high/windows_size)


    for wid_ in range(wid_num):
        for len_ in range(hig_num):
            img_new = img[len_*windows_size:(len_+1)*windows_size, wid_*windows_size:(wid_+1)*windows_size]
            image = Image.fromarray(np.uint8(img_new))
            image.show()


    # skimage.transform.resize(image, [300,300])