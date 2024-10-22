## FLATS: <ins>F</ins>ederated <ins>L</ins>earning <ins>A</ins>dversarial <ins>T</ins>raining for <ins>S</ins>mart Home Face Recognition System

<img align='right' width="350" alt="robust_FL_diagram_noniid" src="https://user-images.githubusercontent.com/56494297/201699788-ea7e5f33-dc48-4c12-a526-a8399c6d1e7e.png">


* This is the official repository of the code of ["Robust Smart Home Face Recognition under Starving Federated Data"](https://arxiv.org/abs/2211.05410) paper, which has been accepted by the [IEEE International Conference on Universal Village (IEEE UV2022)](https://universalvillage.org/).
* We propose a novel robust federated learninng training method for smart home face recognition system named <ins>**FLATS: Federated Learning Adversarial Training for Smart Home Face Recognition System.**</ins>
* For general overview of the training process, take a look at the `notebook/FLATS.ipynb` file.
* The code will be continously updated.
* Oral Presentation Link: [https://youtu.be/Tj9QiJEUBXU](https://youtu.be/Tj9QiJEUBXU)

## LICENSES
- Includes software related under the MIT and Apache 2.0 license

## Run

Arguments to be parsed:
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
CUDA_VISIBLE_DEVICES=0 python main.py \
  --main_folder_path 'pins_face_recognition_105_classes' \ 
  --num_clients 5 \ 
  --train_batch_size 64 \ 
  --test_batch_size 64 \ 
  --num_selected 5 \ 
  --num_attack 1 \ 
  --num_rounds 10 \ 
  --num_local_epochs 5 \ 
  --clean_train_batch_ratio 5 \ 
  --atk FFGSM(white_model, eps=8/255, alpha=10/255)
```




