from PIL import Image
import os

path = './download data'

FileList = os.listdir('%s/Train' % path)

if __name__ == '__main__':

    SizeList = []
    i = 0
    for each in FileList:
        img = Image.open('%s/Train/%s' % (path, each))
        A = list(img.size)  # [792,792]
        A.sort()  # 将[700,800]&[800,700]视为同样大小
        SizeList.append(list(img.size))
        i += 1
        if i % 100 == 0:
            print(i)
    # print(FileList[SizeList.index([90,90])])        # 查询指定大小的图片名
    SizeList.sort()
    print(SizeList)

    ListS = []  # Size
    ListN = []  # Quantity
    j = 0
    k = 1
    while k < i:
        if SizeList[j] == SizeList[k]:
            k += 1
        else:
            ListS.append(SizeList[j])
            ListN.append(k - j)
            j = k
            k += 1
    ListS.append(SizeList[i - 1])  # 列表最后一项（放入while会越界 故单独处理）
    ListN.append(1)
    print(ListS)
    print(ListN)
    n = 0  # 检验结果
    for each in ListN:
        n += each
    print(n)

    T = list(zip(ListS, ListN))
    print(T)
