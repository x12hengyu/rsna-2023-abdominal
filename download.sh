# kaggle datasets download theoviel/rsna-abdominal-trauma-detection-png-pt1 &
kaggle datasets download theoviel/rsna-abdominal-trauma-detection-png-pt2 &
kaggle datasets download theoviel/rsna-2023-abdominal-trauma-detection-pngs-3-8 &
kaggle datasets download theoviel/rsna-abdominal-trauma-detection-png-pt4 &
kaggle datasets download theoviel/rsna-abdominal-trauma-detection-png-pt5 &
kaggle datasets download theoviel/rsna-abdominal-trauma-detection-png-pt6 &
kaggle datasets download theoviel/rsna-abdominal-trauma-detection-pngs-pt7 &
kaggle datasets download theoviel/rsna-2023-abdominal-trauma-detection-pngs-18 &

wait

unzip rsna-abdominal-trauma-detection-png-pt1.zip -d /home/pranav/remote/xizheng/train_images
unzip rsna-abdominal-trauma-detection-png-pt2.zip -d /home/pranav/remote/xizheng/train_images
unzip rsna-2023-abdominal-trauma-detection-pngs-3-8.zip -d /home/pranav/remote/xizheng/train_images
unzip rsna-abdominal-trauma-detection-png-pt4.zip -d /home/pranav/remote/xizheng/train_images
unzip rsna-abdominal-trauma-detection-png-pt5.zip -d /home/pranav/remote/xizheng/train_images
unzip rsna-abdominal-trauma-detection-png-pt6.zip -d /home/pranav/remote/xizheng/train_images
unzip rsna-abdominal-trauma-detection-pngs-pt7.zip -d /home/pranav/remote/xizheng/train_images
unzip rsna-2023-abdominal-trauma-detection-pngs-18.zip -d /home/pranav/remote/xizheng/train_images

wait

# CLEAN UP
rm -rf *.zip
