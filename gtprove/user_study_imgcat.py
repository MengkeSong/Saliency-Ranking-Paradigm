import PIL.Image as Image
import os



IMAGES_FORMAT = ['.png', '.jpg']  # 图片格式
IMAGE_SIZE_width = 640  # 每张小图片的大小
IMAGE_SIZE_height = 480
IMAGE_ROW = 2  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 2  # 图片间隔，也就是合并成一张图后，一共有几列

img_name_index = 1
IMAGE_SAVE_PATH = '/home/w509/桌面/0_5_1_cat/'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称


oriimage_path ='/home/w509/1workspace/lee/360_fix_sort/gtprove/user_study/train100_rank/0_1/'
image_0_1_path = '/home/w509/1workspace/lee/360_fix_sort/gtprove/user_study/train100_rank/0_2/'
image_0_5_path = '/home/w509/1workspace/lee/360_fix_sort/gtprove/user_study/train100_rank/0_5/'
image_1_0_path = '/home/w509/1workspace/lee/360_fix_sort/gtprove/user_study/train100_rank/1_0/'

image_0_1_path_names = os.listdir(image_0_1_path)
image_0_1_path_names.sort(key=lambda x: int(x[:-4]))
image_0_5_path_names = os.listdir(image_0_5_path)
image_0_5_path_names.sort(key=lambda x: int(x[:-4]))
image_1_0_path_names = os.listdir(image_1_0_path)
image_1_0_path_names.sort(key=lambda x: int(x[:-4]))



# 定义图像拼接函数
def image_compose(ori_path,img_0_1_path,img_0_5_path,img_1_0_path,out_path):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE_width, IMAGE_ROW * IMAGE_SIZE_height))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上

    from_image = Image.open(ori_path).resize(
        (IMAGE_SIZE_width, IMAGE_SIZE_height), Image.ANTIALIAS)
    to_image.paste(from_image, ((1 - 1) * IMAGE_SIZE_width, (1 - 1) * IMAGE_SIZE_height))

    from_image = Image.open(img_0_1_path).resize(
        (IMAGE_SIZE_width, IMAGE_SIZE_height), Image.ANTIALIAS)
    to_image.paste(from_image, ((2 - 1) * IMAGE_SIZE_width, (1 - 1) * IMAGE_SIZE_height))

    from_image = Image.open(img_0_5_path).resize(
        (IMAGE_SIZE_width, IMAGE_SIZE_height), Image.ANTIALIAS)
    to_image.paste(from_image, ((1 - 1) * IMAGE_SIZE_width, (2 - 1) * IMAGE_SIZE_height))

    from_image = Image.open(img_1_0_path).resize(
        (IMAGE_SIZE_width, IMAGE_SIZE_height), Image.ANTIALIAS)
    to_image.paste(from_image, ((2 - 1) * IMAGE_SIZE_width, (2 - 1) * IMAGE_SIZE_height))


    # for y in range(1, IMAGE_ROW + 1):
    #     for x in range(1, IMAGE_COLUMN + 1):
    #         from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
    #             (IMAGE_SIZE_width, IMAGE_SIZE_height), Image.ANTIALIAS)
    #         to_image.paste(from_image, ((x - 1) * IMAGE_SIZE_width, (y - 1) * IMAGE_SIZE_height))
    return to_image.save(out_path)  # 保存新图




for index in range(len(image_0_1_path_names)):
    ori_path = oriimage_path+image_0_1_path_names[index]
    img_0_1_path = image_0_1_path+image_0_1_path_names[index]
    img_0_5_path = image_0_5_path+image_0_5_path_names[index]
    img_1_0_path = image_1_0_path+image_1_0_path_names[index]
    out_path = IMAGE_SAVE_PATH+"{:06d}".format(index+1)+".jpg"
    image_compose(ori_path,img_0_1_path,img_0_5_path,img_1_0_path,out_path)





