"""
import math
import sys
import time
import pytesseract as ts
from PIL import Image

ts.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

start = time.time()
a=['0','1','2']
f = open("D:/test/seongtaek.txt",'w')
for i in a:
    filepath = "D:/test/20230102140615_" + i + "_1234.jpeg"
    img = Image.open(filepath)
    text = ts.image_to_string(img, lang='kor')
    f.write(text)
f.close()
end = time.time()
print(end - start)

start1 = time.time()
a1=['0','1','2','3']
f1 = open("D:/test/seongtaek1.txt",'w')
for i in a1:
    filepath1 = "D:/test/20230102143702_" + i + "_NAVER.jpeg"
    img1 = Image.open(filepath1)
    text1 = ts.image_to_string(img1, lang='kor')
    f1.write(text1)
f1.close()
end1 = time.time()
print(end1 - start1)
"""

"""
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
file_name="D:/test/20230102140615_0_1234.jpeg"
east="D:/test/frozen_east_text_detection.pb"
min_confidence=0.5
width=320
height=320


# load the input image and grab the image dimensions
image=cv2.imread(file_name)
orig_image=image.copy()
text_extract_image=image.copy()
(H,W) = image.shape[:2]

# 새로운 width와 height를 설정하고 비율을 구한다
(newW,newH) = width,height
rW=W/float(newW)
rH=H/float(newH)

# image의 size를 재설정하고 새 이미지의 dimension을 구한다
image=cv2.resize(image,(newW,newH))
(H,W) = image.shape[:2]

layerNames=[
    'feature_fusion/Conv_7/Sigmoid',
    'feature_fusion/concat_3'
]

# load the pre-trained EAST text detector
print('[INFO] loading EAST text detector...')
net=cv2.dnn.readNet(east)

blob=cv2.dnn.blobFromImage(image,1.0,(H,W),
    (123.68,116.78,103.94),swapRB=True,crop=False)
start=time.time()
net.setInput(blob)

# geometry는 우리의 input image로 부터 bounding box좌표를 얻게해준다
# scores는 주어진 지역에 text가 있는지에 대한 확률을 준다
(scores,geometry) = net.forward(layerNames)
end=time.time()

print('[INFO] text detection took {:.6f} seconds'.format(end-start))

# scores의 크기를 받고 bounding box 사각형을 추출한뒤 confidencs scores에 대응해본다
(numRows,numCols) = scores.shape[2:4]
rects=[]
confidences=[]
for y in range(0,numRows):
    scoresData=scores[0,0,y]
    xData0=geometry[0,0,y]
    xData1=geometry[0,1,y]
    xData2=geometry[0,2,y]
    xData3=geometry[0,3,y]
    anglesData=geometry[0,4,y]

    for x in range(0,numCols):
        # 만약 score가 충분한 확률을 가지고 있지 않다면 무시한다
        if scoresData[x] < min_confidence:
            continue
      
        # 우리의 resulting feature map은 input_image보다 4배 작을것 이기 때문에
        # offset factor를 계산한다
        (offsetX,offsetY) = (x*4.0,y*4.0)

        # prediciton에 대한 회전각을 구하고 sin,cosine을 계산한다
        # 글씨가 회전되어 있을때를 대비
        angle=anglesData[x]
        cos=np.cos(angle)
        sin=np.sin(angle)

        # geometry volume를 사용해 bounding box의 width 와 height를 구한다
        h=xData0[x] + xData2[x]
        w=xData1[x] + xData3[x]

        # text prediction bounding box의 starting, ending (x,y) 좌표를 계산한다
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)
      
        # bounding box coordinates와 probability score를 append한다
        rects.append((startX,startY,endX,endY))
        confidences.append(scoresData[x])

# non-maxima suppression 을 weak,overlapping bounding boxes을 없애기위해 적용해준다
boxes=non_max_suppression(np.array(rects),probs=confidences)

def textRead(image):
    # apply Tesseract v4 to OCR 
    config = ("-l eng+kor --oem 1 --psm 7")
    text = pytesseract.image_to_string(image, config=config)
    # display the text OCR'd by Tesseract
    print("OCR TEXT : {}".format(text))
  
    # strip out non-ASCII text 
    #text = "".join([c if c.isalnum() else "" for c in text]).strip()
    #print("Alpha numeric TEXT : {}\n".format(text))
    return text

image2 = orig_image.copy()
i = 0

for (startX,startY,endX,endY) in boxes:
    # 앞에서 구한 비율에 따라서 bounding box 좌표를 키워준다
    startX=int(startX * rW)
    startY=int(startY * rH)
    endX=int(endX * rW)
    endY=int(endY * rH)
  
    text=textRead(text_extract_image[startY:endY, startX:endX])
    #cropped = image2[startY:endY, startX:endX]
    #cv2.imwrite("D:/test/crop_img_{}.jpg".format(i), cropped)
    #i = i+1
    cv2.rectangle(orig_image,(startX,startY),(endX,endY),(0,255,0),2)
    cv2.putText(orig_image, text, (startX, startY-10),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

#cv2.imshow('Text Detection', orig_image)
cv2.imwrite("D:/test/ost.png", orig_image)
cv2.waitKey(0)

"""

 # importing modules
import cv2
import pytesseract
import csv
import time
import sys

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

start = time.time()

image = cv2.imread("D:/Git/OpenCV/testfile/20230102143702_0_NAVER.jpeg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#cv2.imwrite("D:/Git/OpenCV/testfile/12345.png", threshold_img)

custom_config = r'--oem 3 --psm 6'
details = pytesseract.image_to_data(threshold_img, output_type=pytesseract.Output.DICT, config=custom_config, lang='eng+kor')

total_boxes = len(details['text'])

'''
# 문자열 어떻게 스트링화 할지 나타내는 좌표 이미지 
for sequence_number in range(total_boxes):
	if int(details['conf'][sequence_number]) >30:
		(x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])
		threshold_img = cv2.rectangle(threshold_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# display image
cv2.imwrite("D:/Git/OpenCV/testfile/12345_go.png", threshold_img)
'''
parse_text = []
word_list = []
last_word = ''

for word in details['text']:
    if word!='':
        word_list.append(word)
        last_word = word
    if (last_word!='' and word == '') or (word==details['text'][-1]):
        parse_text.append(word_list)
        word_list = []
 
with open('D:/Git/OpenCV/testfile/12345_go123.txt',  'w', newline="", encoding='UTF-8') as file:
    csv.writer(file, delimiter=" ").writerows(parse_text)

end = time.time()
print(end-start)
