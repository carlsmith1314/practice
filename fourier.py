import cv2
import numpy as np

src_image = cv2.imread('20190417104624678.PNG')


cv2.namedWindow("yuan", cv2.WINDOW_NORMAL)
cv2.imshow("yuan", src_image)


blur_image = cv2.GaussianBlur(src_image, (15, 15), 0, 0)
cv2.namedWindow("GUSS", cv2.WINDOW_NORMAL)
cv2.imshow("GUSS", blur_image)


gray_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
ret, binary_image = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
cv2.imshow("gray", binary_image)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, (-1, -1))
cv2.namedWindow("xingtaixue", cv2.WINDOW_NORMAL)
cv2.imshow("xingtaixue", morph_image)

print(src_image.shape)
result_image = np.zeros(src_image.shape, np.float32)
contours, hier = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(result_image, contours, -1, (255, 255, 255), 1)
cv2.imshow("lunkuo", result_image)

index = 0
data = []
while(index < len(contours[0])):
    data.append(complex(contours[0][index][0][0], contours[0][index][0][1]))
    index = index + 1


dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)
result = 20 * np.log(cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1]))
print(result)


index1 = 0
fourier = []
while(index1 < len(result[0])):
    if(index1 == 0):
        fourier.append(result[0])
    else:
        fourier.append(result[index1]/result[1])
    index1 = index1 + 1
print(fourier)
cv2.waitKey()
