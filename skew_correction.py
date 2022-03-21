import os, sys
import cv2
import numpy as np
from scipy.ndimage import rotate as rotate_scipy
from PIL import Image
import pandas as pd
from skimage.transform import rotate
from skimage import io
import time
from deskew import determine_skew
start_time = time.time()



class SkewCorrection():
    def correct_skew(self, image_path, delta=1, limit=5):
        im = cv2.imread(image_path)
        grayscale = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        start_time = time.time()
        angle = determine_skew(grayscale)
        if angle != 0:
            height, width = im.shape[:2]
            center_img = (width / 2, height / 2)
            rotationMatrix = cv2.getRotationMatrix2D(center_img, angle, 1.0)
            rotated_img = cv2.warpAffine(im, rotationMatrix, (width, height), borderMode = cv2.BORDER_REFLECT)
            rotated = rotate(im, angle, resize=True) * 255
            #io.imsave('output.png', rotated.astype(np.uint8))
            rgb_image = cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            f = os.path.splitext(image_path)[0]
            pil_image.save("%s_rotated_%s.png"%(f,angle), dpi = (300,300))
            #pil_image.save(image_path, dpi = (300,300))
        return angle

if __name__ == '__main__':
    cl = SkewCorrection()
    img_file = sys.argv[1]
    files = os.listdir(img_file)
    skew_test = pd.DataFrame(columns = ['file_name', 'original_angle', 'result_angle', 'difference', 'match'])
    for f_name in files:
        original_angle = f_name.split("_")[0]
        print("original",original_angle)
        start_time_1image = time.time()
        image_path = "%s\%s"%(img_file, f_name)
        res_angle = cl.correct_skew(image_path)
        diff = abs(abs(int(original_angle)) - abs(float(res_angle)))
        match = False
        if diff <=2:
            print("name",f_name)
            match = True
        skew_test.loc[len(skew_test.index)] = [f_name, original_angle, res_angle, diff, match]
        print("res angle", res_angle)

        #print(f_name,"--- %s seconds ---" % (time.time() - start_time_1image))
    skew_test.to_excel("skew_test.xlsx")
    print("--- %s seconds ---" % (time.time() - start_time))
