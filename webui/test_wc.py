from weapons_classifier import WeaponsClassifier
import cv2


test_img = cv2.imread("/home/hbdesk/labelstudio_convert/weapon_img/AWM/08dbec4c-3087f791e5674b6885225e58e02a184c.png")

wc = WeaponsClassifier("../weapons_classifier/weapon_classifier.pth")
results = wc.get_weapon_class(test_img)
print(results)
