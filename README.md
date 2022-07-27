# Robust Federated Learning in Smart Home Face Recognition System

* The purpose of this project is to experiment the robustness of federated learning in smart home face recognition system.
* The code will be continously updated.

## LICENSES
- Includes software related under the MIT and Apache 2.0 license

## Run

Arguments that need to be parsed:
* --main_folder_path             
* --num_clients                  (default=5) 
* --train_batch_size             (default=64)
* --test_batch_size              (default=64)
* --num_selected                 (default=5)
* --num_attack                   (default=1)
* --num_rounds                   (default=10)
* --num_local_epochs             (default=5)
* --clean_train_batch_ratio      (default=5)
* --atk                          (default=FFGSM(white_model, eps=8/255, alpha=10/255))

Run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --main_folder_path 'pins_face_recognition_105_classes' --num_clients 5 --train_batch_size 64 --test_batch_size 64 --num_selected 5 --num_attack 1 --num_rounds 10 --num_local_epochs 5 --clean_train_batch_ratio 5 --atk FFGSM(white_model, eps=8/255, alpha=10/255)
```




