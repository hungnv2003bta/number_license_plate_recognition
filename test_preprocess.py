import cv2

# Load an image
img_name = "AQUA2_54827_checkin_2020-10-31-11-36MomsrrHfJ2.jpg"
dataset_label_path = "./dataset/test/new_test.csv"
dataset_img_path = "./dataset/test/test/"
img_path = dataset_img_path + img_name
image = cv2.imread(img_path)

# Experiment with different parameters
params = [
    (9, 75, 75),  # Default-like
    (9, 100, 100),  # Stronger noise reduction
    (5, 50, 50)  # Fine details preservation
]

for i, (d, sigmaColor, sigmaSpace) in enumerate(params):
    filtered_image = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    cv2.imshow(f'Bilateral Filter - Params {i+1}', filtered_image)

cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
