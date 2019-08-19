## Face image 
##### need to create a directory for kep face image 
    face_image
        --->test
            ---> 1.jpg
        --->train
            ---> borhan
                    ---> img1.jpg
                    ---> img2.jpg
                    --->
                    ---> imgn.jpg
            ---> raihan
                    ---> img1.jpg
                    ---> img2.jpg
                    --->
                    ---> imgn.jpg
## Run

    python3 train.py
    
## Test 

    python3 test.py    
    
    
## Crop face 
    
    cd align_all
    python3 face_align.py -source_root /home/borhan/face.evoLVe.PyTorch/data/imgs/test -dest_root /home/borhan/face.evoLVe.PyTorch/data/imgs/test_Aligned -crop_size 112
    python face_align.py -source_root E:\Project\KNN\final_face_recognition\train_face\face_image\train -dest_root E:\Project\KNN\final_face_recognition\train_face\face_image\test_Aligned -crop_size 112
#### Reference
https://github.com/ZhaoJ9014/face.evoLVe.PyTorch                                   