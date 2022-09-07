# Projeto Final - Modelos Preditivos Conexionistas

### Eden Coelho (ecdf@cesar.shool)

|**Classificação de imagens**|**YOLOV5**|**PYTHON**|
|--|--|--|
|Classificação de Imagens<br>ou<br>Deteção de Objetos| YOLOv5

## Performance

Your dataset_imagens (2022-08-20 8:53am) model has finished training in 12 minutes and achieved 99.5% mAP , 69.8% precision , and 100.0% recall .

### Output do bloco de treinamento

<details>
  <summary>Click to expand!</summary>
  
  ```
  train: weights=yolov5s.pt, cfg=, data=/content/datasets/dataset_imagens-3/data.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=150, batch_size=16, imgsz=416, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v6.2-96-g5a134e0 Python-3.7.13 torch-1.12.1+cu113 CUDA:0 (Tesla T4, 15110MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Weights & Biases: run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs in Weights & Biases
ClearML: run 'pip install clearml' to automatically track, visualize and remotely train YOLOv5 🚀 in ClearML
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Overriding model.yaml nc=80 with nc=3

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     21576  models.yolo.Detect                      [3, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 270 layers, 7027720 parameters, 7027720 gradients, 16.0 GFLOPs

Transferred 343/349 items from yolov5s.pt
AMP: checks passed ✅
optimizer: SGD(lr=0.01) with parameter groups 57 weight(decay=0.0), 60 weight(decay=0.0005), 60 bias
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
train: Scanning '/content/datasets/dataset_imagens-3/train/labels' images and labels...11 found, 0 missing, 0 empty, 0 corrupt: 100% 11/11 [00:00<00:00, 1581.24it/s]
train: New cache created: /content/datasets/dataset_imagens-3/train/labels.cache
train: Caching images (0.0GB ram): 100% 11/11 [00:00<00:00, 448.91it/s]
val: Scanning '/content/datasets/dataset_imagens-3/valid/labels' images and labels...3 found, 0 missing, 0 empty, 0 corrupt: 100% 3/3 [00:00<00:00, 626.20it/s]
val: New cache created: /content/datasets/dataset_imagens-3/valid/labels.cache
val: Caching images (0.0GB ram): 100% 3/3 [00:00<00:00, 152.94it/s]

AutoAnchor: 2.45 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Plotting labels to runs/train/exp2/labels.jpg... 
Image sizes 416 train, 416 val
Using 2 dataloader workers
Logging results to runs/train/exp2
Starting training for 150 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      0/149      4.67G     0.1171    0.02023    0.03819         30        416: 100% 1/1 [00:03<00:00,  3.27s/it]
/usr/local/lib/python3.7/dist-packages/torch/optim/lr_scheduler.py:136: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  3.19it/s]
                   all          3          3    0.00553          1     0.0854     0.0229

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      1/149       4.7G     0.1184    0.01882    0.03897         27        416: 100% 1/1 [00:00<00:00,  7.93it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  8.06it/s]
                   all          3          3    0.00559          1      0.172     0.0318

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      2/149      4.71G    0.07622    0.02099    0.02832         32        416: 100% 1/1 [00:00<00:00,  7.65it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.77it/s]
                   all          3          3     0.0056          1      0.149      0.029

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      3/149      4.71G    0.07218    0.02178    0.02617         33        416: 100% 1/1 [00:00<00:00,  7.63it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 18.07it/s]
                   all          3          3    0.00561          1      0.162     0.0292

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      4/149      4.71G     0.1215    0.02284    0.04401         34        416: 100% 1/1 [00:00<00:00,  6.99it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 18.92it/s]
                   all          3          3    0.00552          1     0.0634     0.0205

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      5/149      4.71G     0.1204    0.02099    0.03812         31        416: 100% 1/1 [00:00<00:00,  7.65it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 18.52it/s]
                   all          3          3    0.00552          1      0.163     0.0292

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      6/149      4.71G     0.1104    0.01812    0.04314         27        416: 100% 1/1 [00:00<00:00,  7.39it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 15.60it/s]
                   all          3          3    0.00545          1     0.0444      0.015

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      7/149      4.71G    0.06682     0.0214    0.02927         29        416: 100% 1/1 [00:00<00:00,  7.25it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 20.68it/s]
                   all          3          3     0.0055          1     0.0651     0.0258

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      8/149      4.71G    0.07078    0.01877    0.02709         26        416: 100% 1/1 [00:00<00:00,  7.37it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 21.56it/s]
                   all          3          3    0.00546          1      0.114     0.0352

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      9/149      4.71G       0.12    0.02113    0.04275         30        416: 100% 1/1 [00:00<00:00,  7.16it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 19.45it/s]
                   all          3          3    0.00521          1       0.13     0.0545

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     10/149      4.71G     0.1191     0.0236    0.03901         35        416: 100% 1/1 [00:00<00:00,  6.81it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 19.40it/s]
                   all          3          3    0.00502          1      0.305      0.124

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     11/149      4.71G    0.06829    0.01978      0.026         26        416: 100% 1/1 [00:00<00:00,  6.22it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00,  7.49it/s]
                   all          3          3    0.00512          1      0.304      0.155

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     12/149      4.71G     0.0687    0.01905      0.026         26        416: 100% 1/1 [00:00<00:00,  7.29it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 19.61it/s]
                   all          3          3    0.00519          1      0.217      0.105

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     13/149      4.71G    0.09886    0.02073    0.04709         27        416: 100% 1/1 [00:00<00:00,  6.61it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 20.63it/s]
                   all          3          3    0.00531          1       0.21      0.126

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     14/149      4.71G      0.066    0.02301    0.02589         33        416: 100% 1/1 [00:00<00:00,  7.44it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 21.64it/s]
                   all          3          3    0.00526          1      0.181      0.102

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     15/149      4.71G     0.1075    0.02364    0.04827         30        416: 100% 1/1 [00:00<00:00,  7.41it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 21.05it/s]
                   all          3          3    0.00496          1      0.199     0.0912

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     16/149      4.71G     0.1032    0.01981    0.04059         24        416: 100% 1/1 [00:00<00:00,  7.78it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 22.40it/s]
                   all          3          3    0.00502          1      0.198     0.0854

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     17/149      4.71G    0.05902    0.01777    0.02735         21        416: 100% 1/1 [00:00<00:00, 10.49it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 19.16it/s]
                   all          3          3    0.00502          1      0.198     0.0854

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     18/149      4.71G    0.06098    0.02431    0.02618         30        416: 100% 1/1 [00:00<00:00,  7.85it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.15it/s]
                   all          3          3    0.00502          1      0.167     0.0679

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     19/149      4.71G    0.05778    0.02319    0.02588         30        416: 100% 1/1 [00:00<00:00,  9.34it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 22.67it/s]
                   all          3          3    0.00502          1      0.167     0.0679

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     20/149      4.71G    0.05755    0.01954    0.02623         25        416: 100% 1/1 [00:00<00:00,  7.16it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.20it/s]
                   all          3          3    0.00494          1      0.169     0.0689

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     21/149      4.71G     0.0593    0.02721    0.02589         33        416: 100% 1/1 [00:00<00:00,  8.44it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 20.52it/s]
                   all          3          3    0.00494          1      0.169     0.0689

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     22/149      4.71G    0.06134    0.01963    0.02602         25        416: 100% 1/1 [00:00<00:00,  7.87it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 22.72it/s]
                   all          3          3    0.00502          1      0.321     0.0882

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     23/149      4.71G    0.09831    0.02676     0.0369         32        416: 100% 1/1 [00:00<00:00,  9.82it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 21.32it/s]
                   all          3          3    0.00502          1      0.321     0.0882

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     24/149      4.71G    0.05733    0.02121    0.02636         27        416: 100% 1/1 [00:00<00:00,  8.27it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.83it/s]
                   all          3          3    0.00502          1      0.324      0.104

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     25/149      4.71G    0.09836    0.02557    0.03917         34        416: 100% 1/1 [00:00<00:00, 10.14it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 19.05it/s]
                   all          3          3    0.00502          1      0.324      0.104

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     26/149      4.71G    0.08459    0.02227    0.03687         25        416: 100% 1/1 [00:00<00:00,  6.36it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.81it/s]
                   all          3          3    0.00506          1      0.338      0.107

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     27/149      4.71G    0.05427    0.02187    0.02549         25        416: 100% 1/1 [00:00<00:00, 10.13it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 21.67it/s]
                   all          3          3    0.00506          1      0.338      0.107

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     28/149      4.71G    0.05584    0.02644    0.02569         32        416: 100% 1/1 [00:00<00:00,  7.07it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 22.83it/s]
                   all          3          3    0.00503          1      0.374      0.132

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     29/149      4.71G    0.09453    0.02538    0.03762         33        416: 100% 1/1 [00:00<00:00,  9.34it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 21.96it/s]
                   all          3          3    0.00503          1      0.374      0.132

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     30/149      4.71G    0.05398    0.02333    0.02441         29        416: 100% 1/1 [00:00<00:00,  6.52it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.83it/s]
                   all          3          3    0.00505          1      0.376       0.13

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     31/149      4.71G    0.09776    0.02394    0.03561         29        416: 100% 1/1 [00:00<00:00,  9.88it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 22.20it/s]
                   all          3          3    0.00505          1      0.376       0.13

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     32/149      4.71G    0.05124    0.02317    0.02543         26        416: 100% 1/1 [00:00<00:00,  8.20it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.63it/s]
                   all          3          3    0.00513          1      0.422      0.157

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     33/149      4.71G    0.04973    0.02799    0.02506         32        416: 100% 1/1 [00:00<00:00, 10.07it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 18.54it/s]
                   all          3          3    0.00513          1      0.422      0.157

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     34/149      4.71G    0.07812    0.02945    0.04073         35        416: 100% 1/1 [00:00<00:00,  7.22it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 22.54it/s]
                   all          3          3    0.00521          1      0.535       0.16

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     35/149      4.71G    0.05184    0.02549    0.02576         30        416: 100% 1/1 [00:00<00:00, 10.40it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 20.88it/s]
                   all          3          3    0.00521          1      0.535       0.16

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     36/149      4.71G    0.08438    0.02174      0.045         24        416: 100% 1/1 [00:00<00:00,  6.71it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.43it/s]
                   all          3          3     0.0603       0.25       0.41      0.142

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     37/149      4.71G     0.0746    0.02997    0.03688         35        416: 100% 1/1 [00:00<00:00,  9.24it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 22.80it/s]
                   all          3          3     0.0603       0.25       0.41      0.142

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     38/149      4.71G    0.07777    0.02634    0.03619         32        416: 100% 1/1 [00:00<00:00,  6.86it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.88it/s]
                   all          3          3      0.604       0.25       0.39      0.138

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     39/149      4.71G    0.04568    0.02382    0.02395         25        416: 100% 1/1 [00:00<00:00,  9.63it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.35it/s]
                   all          3          3      0.604       0.25       0.39      0.138

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     40/149      4.71G      0.095    0.02539    0.03535         30        416: 100% 1/1 [00:00<00:00,  6.74it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.22it/s]
                   all          3          3      0.113      0.925      0.611      0.105

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     41/149      4.71G    0.09437    0.02522    0.03641         31        416: 100% 1/1 [00:00<00:00,  8.43it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.51it/s]
                   all          3          3      0.113      0.925      0.611      0.105

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     42/149      4.71G    0.06377    0.01951    0.05063         22        416: 100% 1/1 [00:00<00:00,  6.14it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.15it/s]
                   all          3          3    0.00589          1       0.32       0.15

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     43/149      4.71G    0.04551    0.03035    0.02504         34        416: 100% 1/1 [00:00<00:00, 10.02it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.21it/s]
                   all          3          3    0.00589          1       0.32       0.15

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     44/149      4.71G    0.09676    0.02636    0.04042         31        416: 100% 1/1 [00:00<00:00,  7.04it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 22.33it/s]
                   all          3          3      0.171          1      0.402      0.172

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     45/149      4.71G    0.05775    0.02242    0.04342         26        416: 100% 1/1 [00:00<00:00,  7.98it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.90it/s]
                   all          3          3      0.171          1      0.402      0.172

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     46/149      4.71G    0.08041    0.02155    0.03646         24        416: 100% 1/1 [00:00<00:00,  6.14it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.36it/s]
                   all          3          3    0.00583          1      0.348     0.0763

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     47/149      4.71G    0.06729      0.027     0.0341         31        416: 100% 1/1 [00:00<00:00,  9.87it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.85it/s]
                   all          3          3    0.00583          1      0.348     0.0763

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     48/149      4.71G    0.04687    0.02105    0.02452         23        416: 100% 1/1 [00:00<00:00,  7.17it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.61it/s]
                   all          3          3    0.00583          1      0.415      0.133

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     49/149      4.71G    0.08961    0.02536    0.04122         29        416: 100% 1/1 [00:00<00:00,  8.13it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.38it/s]
                   all          3          3    0.00583          1      0.415      0.133

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     50/149      4.71G    0.04524    0.02652    0.02434         30        416: 100% 1/1 [00:00<00:00,  7.68it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.24it/s]
                   all          3          3    0.00573          1      0.497      0.176

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     51/149      4.71G    0.04054    0.02224    0.02323         25        416: 100% 1/1 [00:00<00:00,  9.98it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.79it/s]
                   all          3          3    0.00573          1      0.497      0.176

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     52/149      4.71G    0.09912    0.02477    0.03255         31        416: 100% 1/1 [00:00<00:00,  8.06it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.80it/s]
                   all          3          3    0.00573          1      0.497      0.176

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     53/149      4.71G    0.04528    0.02495    0.02251         30        416: 100% 1/1 [00:00<00:00,  7.15it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.66it/s]
                   all          3          3    0.00577          1      0.697      0.156

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     54/149      4.71G     0.0437    0.02623     0.0235         30        416: 100% 1/1 [00:00<00:00,  8.66it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.35it/s]
                   all          3          3    0.00577          1      0.697      0.156

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     55/149      4.71G    0.04674    0.02278    0.02381         28        416: 100% 1/1 [00:00<00:00,  9.47it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.47it/s]
                   all          3          3    0.00577          1      0.697      0.156

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     56/149      4.71G     0.0411    0.01812    0.02282         21        416: 100% 1/1 [00:00<00:00,  7.94it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.80it/s]
                   all          3          3    0.00845          1       0.87      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     57/149      4.71G    0.06846    0.02821    0.03615         34        416: 100% 1/1 [00:00<00:00, 10.05it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 22.29it/s]
                   all          3          3    0.00845          1       0.87      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     58/149      4.71G    0.07766    0.02404    0.03515         29        416: 100% 1/1 [00:00<00:00,  9.60it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.94it/s]
                   all          3          3    0.00845          1       0.87      0.248

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     59/149      4.71G    0.06199     0.0232     0.0364         26        416: 100% 1/1 [00:00<00:00,  7.12it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.37it/s]
                   all          3          3     0.0057          1      0.746      0.283

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     60/149      4.71G     0.0472    0.02389    0.02339         29        416: 100% 1/1 [00:00<00:00,  8.34it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 19.57it/s]
                   all          3          3     0.0057          1      0.746      0.283

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     61/149      4.71G    0.07652    0.02485    0.03898         30        416: 100% 1/1 [00:00<00:00,  9.80it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.41it/s]
                   all          3          3     0.0057          1      0.746      0.283

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     62/149      4.71G    0.06199    0.02556    0.03266         30        416: 100% 1/1 [00:00<00:00,  7.47it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.56it/s]
                   all          3          3     0.0101          1      0.497      0.262

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     63/149      4.71G    0.05004    0.02499    0.03236         28        416: 100% 1/1 [00:00<00:00,  7.82it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.76it/s]
                   all          3          3     0.0101          1      0.497      0.262

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     64/149      4.71G    0.07811    0.02531    0.04191         32        416: 100% 1/1 [00:00<00:00,  9.28it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.83it/s]
                   all          3          3     0.0101          1      0.497      0.262

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     65/149      4.71G     0.0557    0.02118    0.06335         25        416: 100% 1/1 [00:00<00:00,  6.87it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.32it/s]
                   all          3          3     0.0772          1      0.912       0.49

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     66/149      4.71G    0.06466    0.02875    0.03582         35        416: 100% 1/1 [00:00<00:00, 10.11it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.74it/s]
                   all          3          3     0.0772          1      0.912       0.49

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     67/149      4.71G    0.07163    0.02728    0.03957         32        416: 100% 1/1 [00:00<00:00,  7.95it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.50it/s]
                   all          3          3     0.0772          1      0.912       0.49

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     68/149      4.71G    0.04472    0.02084    0.02293         25        416: 100% 1/1 [00:00<00:00,  8.25it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.03it/s]
                   all          3          3      0.216          1      0.912      0.474

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     69/149      4.71G    0.04071     0.0287    0.02174         33        416: 100% 1/1 [00:00<00:00,  9.08it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.23it/s]
                   all          3          3      0.216          1      0.912      0.474

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     70/149      4.71G     0.0647    0.02404    0.03825         30        416: 100% 1/1 [00:00<00:00,  9.50it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.61it/s]
                   all          3          3      0.216          1      0.912      0.474

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     71/149      4.71G    0.09226    0.02511    0.03579         31        416: 100% 1/1 [00:00<00:00,  7.20it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.13it/s]
                   all          3          3      0.144          1      0.912      0.462

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     72/149      4.71G    0.04504    0.02179     0.0236         29        416: 100% 1/1 [00:00<00:00,  9.90it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.86it/s]
                   all          3          3      0.144          1      0.912      0.462

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     73/149      4.71G    0.03853    0.01747    0.02504         20        416: 100% 1/1 [00:00<00:00, 10.23it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.13it/s]
                   all          3          3      0.144          1      0.912      0.462

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     74/149      4.71G    0.07417    0.02216    0.03355         29        416: 100% 1/1 [00:00<00:00,  6.11it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.97it/s]
                   all          3          3      0.745        0.5      0.912      0.394

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     75/149      4.71G    0.07523    0.02933     0.0367         35        416: 100% 1/1 [00:00<00:00,  9.02it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.70it/s]
                   all          3          3      0.745        0.5      0.912      0.394

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     76/149      4.71G    0.04157    0.02677    0.02279         31        416: 100% 1/1 [00:00<00:00,  6.01it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.12it/s]
                   all          3          3      0.745        0.5      0.912      0.394

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     77/149      4.71G    0.05719    0.02672     0.0296         31        416: 100% 1/1 [00:00<00:00,  7.02it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.47it/s]
                   all          3          3      0.724      0.423       0.87      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     78/149      4.71G    0.05166     0.0269    0.03627         32        416: 100% 1/1 [00:00<00:00,  9.43it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.07it/s]
                   all          3          3      0.724      0.423       0.87      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     79/149      4.71G    0.04906    0.02626    0.04564         31        416: 100% 1/1 [00:00<00:00,  8.89it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.15it/s]
                   all          3          3      0.724      0.423       0.87      0.334

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     80/149      4.71G    0.05591    0.02359    0.03341         28        416: 100% 1/1 [00:00<00:00,  6.80it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.57it/s]
                   all          3          3      0.794        0.5      0.912      0.397

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     81/149      4.71G    0.03831    0.02263    0.02111         27        416: 100% 1/1 [00:00<00:00,  8.64it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.99it/s]
                   all          3          3      0.794        0.5      0.912      0.397

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     82/149      4.71G    0.07239    0.03056     0.0381         36        416: 100% 1/1 [00:00<00:00,  8.54it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.08it/s]
                   all          3          3      0.794        0.5      0.912      0.397

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     83/149      4.71G    0.04104     0.0248    0.02204         32        416: 100% 1/1 [00:00<00:00,  8.38it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 28.13it/s]
                   all          3          3      0.793        0.5      0.995      0.398

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     84/149      4.71G    0.04035    0.02183    0.02257         27        416: 100% 1/1 [00:00<00:00,  9.34it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.54it/s]
                   all          3          3      0.793        0.5      0.995      0.398

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     85/149      4.71G    0.04039    0.01833    0.02155         25        416: 100% 1/1 [00:00<00:00,  7.84it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.31it/s]
                   all          3          3      0.793        0.5      0.995      0.398

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     86/149      4.71G    0.05771    0.02172    0.04441         27        416: 100% 1/1 [00:00<00:00,  9.05it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.99it/s]
                   all          3          3      0.793        0.5      0.995      0.398

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     87/149      4.71G    0.06229    0.01889    0.03912         22        416: 100% 1/1 [00:00<00:00,  6.23it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.66it/s]
                   all          3          3      0.496          1      0.912      0.497

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     88/149      4.71G    0.06784    0.02922    0.03769         35        416: 100% 1/1 [00:00<00:00,  8.07it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.35it/s]
                   all          3          3      0.496          1      0.912      0.497

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     89/149      4.71G    0.04123    0.02702    0.02224         35        416: 100% 1/1 [00:00<00:00,  8.92it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.92it/s]
                   all          3          3      0.496          1      0.912      0.497

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     90/149      4.71G    0.06877    0.02629    0.02908         31        416: 100% 1/1 [00:00<00:00,  8.12it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.42it/s]
                   all          3          3      0.496          1      0.912      0.497

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     91/149      4.71G    0.04126    0.02647    0.02178         32        416: 100% 1/1 [00:00<00:00,  7.56it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 28.45it/s]
                   all          3          3      0.317          1      0.829      0.423

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     92/149      4.71G    0.03669    0.01979    0.02291         25        416: 100% 1/1 [00:00<00:00,  9.91it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.12it/s]
                   all          3          3      0.317          1      0.829      0.423

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     93/149      4.71G    0.08352    0.02183    0.03472         28        416: 100% 1/1 [00:00<00:00,  8.84it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.33it/s]
                   all          3          3      0.317          1      0.829      0.423

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     94/149      4.71G    0.07772    0.02568    0.03693         32        416: 100% 1/1 [00:00<00:00,  7.36it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.71it/s]
                   all          3          3      0.317          1      0.829      0.423

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     95/149      4.71G    0.05688    0.02233    0.03761         29        416: 100% 1/1 [00:00<00:00,  6.49it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.37it/s]
                   all          3          3      0.533          1      0.912      0.473

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     96/149      4.71G    0.06091    0.02459    0.03251         30        416: 100% 1/1 [00:00<00:00,  8.92it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.81it/s]
                   all          3          3      0.533          1      0.912      0.473

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     97/149      4.71G     0.0379    0.02162    0.02016         29        416: 100% 1/1 [00:00<00:00,  9.71it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.78it/s]
                   all          3          3      0.533          1      0.912      0.473

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     98/149      4.71G    0.05114     0.0246    0.03495         32        416: 100% 1/1 [00:00<00:00,  9.01it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.17it/s]
                   all          3          3      0.533          1      0.912      0.473

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
     99/149      4.71G    0.03942    0.01963    0.02013         25        416: 100% 1/1 [00:00<00:00,  7.79it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.09it/s]
                   all          3          3      0.565      0.993      0.912      0.555

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    100/149      4.71G    0.03623    0.02169    0.02191         28        416: 100% 1/1 [00:00<00:00,  9.52it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.28it/s]
                   all          3          3      0.565      0.993      0.912      0.555

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    101/149      4.71G    0.07458    0.02469    0.03131         32        416: 100% 1/1 [00:00<00:00,  8.84it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.94it/s]
                   all          3          3      0.565      0.993      0.912      0.555

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    102/149      4.71G    0.03618    0.01667    0.02052         22        416: 100% 1/1 [00:00<00:00,  8.99it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.42it/s]
                   all          3          3      0.565      0.993      0.912      0.555

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    103/149      4.71G    0.03896    0.01893    0.02064         24        416: 100% 1/1 [00:00<00:00,  5.00it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.76it/s]
                   all          3          3      0.611      0.952      0.912      0.551

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    104/149      4.71G    0.03613    0.02074    0.02092         27        416: 100% 1/1 [00:00<00:00,  9.81it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.75it/s]
                   all          3          3      0.611      0.952      0.912      0.551

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    105/149      4.71G    0.06651     0.0271    0.04189         34        416: 100% 1/1 [00:00<00:00,  8.61it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.49it/s]
                   all          3          3      0.611      0.952      0.912      0.551

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    106/149      4.71G    0.03736    0.01807     0.0211         26        416: 100% 1/1 [00:00<00:00,  7.58it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.64it/s]
                   all          3          3      0.611      0.952      0.912      0.551

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    107/149      4.71G    0.03981    0.02416    0.02245         32        416: 100% 1/1 [00:00<00:00,  7.25it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.96it/s]
                   all          3          3      0.541       0.75       0.87      0.555

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    108/149      4.71G    0.03757    0.02586    0.02278         33        416: 100% 1/1 [00:00<00:00,  7.48it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.66it/s]
                   all          3          3      0.541       0.75       0.87      0.555

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    109/149      4.71G     0.0663    0.02292    0.03654         30        416: 100% 1/1 [00:00<00:00,  7.66it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.98it/s]
                   all          3          3      0.541       0.75       0.87      0.555

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    110/149      4.71G    0.03697    0.02493    0.02379         32        416: 100% 1/1 [00:00<00:00,  7.50it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.12it/s]
                   all          3          3      0.541       0.75       0.87      0.555

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    111/149      4.71G    0.04982    0.02458    0.04274         33        416: 100% 1/1 [00:00<00:00,  7.60it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 22.81it/s]
                   all          3          3      0.633      0.923      0.829      0.541

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    112/149      4.71G    0.06451    0.02669    0.02921         35        416: 100% 1/1 [00:00<00:00,  9.83it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.99it/s]
                   all          3          3      0.633      0.923      0.829      0.541

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    113/149      4.71G    0.03598     0.0174    0.02104         24        416: 100% 1/1 [00:00<00:00,  7.53it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 22.57it/s]
                   all          3          3      0.633      0.923      0.829      0.541

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    114/149      4.71G    0.03917    0.01672     0.0193         24        416: 100% 1/1 [00:00<00:00,  9.04it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.23it/s]
                   all          3          3      0.633      0.923      0.829      0.541

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    115/149      4.71G    0.03658    0.02691    0.02145         34        416: 100% 1/1 [00:00<00:00,  7.82it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.31it/s]
                   all          3          3       0.61       0.98      0.829      0.539

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    116/149      4.71G    0.03591     0.0255    0.02096         34        416: 100% 1/1 [00:00<00:00,  9.45it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.42it/s]
                   all          3          3       0.61       0.98      0.829      0.539

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    117/149      4.71G    0.03644     0.0243     0.0182         33        416: 100% 1/1 [00:00<00:00, 10.16it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.43it/s]
                   all          3          3       0.61       0.98      0.829      0.539

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    118/149      4.71G    0.03511    0.02233    0.02252         28        416: 100% 1/1 [00:00<00:00,  9.14it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.20it/s]
                   all          3          3       0.61       0.98      0.829      0.539

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    119/149      4.71G    0.03476    0.02423    0.01817         31        416: 100% 1/1 [00:00<00:00,  8.07it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.83it/s]
                   all          3          3      0.532          1      0.912      0.589

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    120/149      4.71G    0.04984    0.01858    0.04251         23        416: 100% 1/1 [00:00<00:00,  9.81it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.11it/s]
                   all          3          3      0.532          1      0.912      0.589

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    121/149      4.71G    0.03597    0.02533     0.0197         32        416: 100% 1/1 [00:00<00:00, 10.36it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 16.81it/s]
                   all          3          3      0.532          1      0.912      0.589

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    122/149      4.71G    0.03436    0.02074    0.01869         29        416: 100% 1/1 [00:00<00:00, 10.32it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.46it/s]
                   all          3          3      0.532          1      0.912      0.589

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    123/149      4.71G    0.07036    0.02036    0.03416         30        416: 100% 1/1 [00:00<00:00,  7.29it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.28it/s]
                   all          3          3      0.459          1      0.995      0.618

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    124/149      4.71G    0.03667    0.01954    0.02159         24        416: 100% 1/1 [00:00<00:00,  7.94it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.79it/s]
                   all          3          3      0.459          1      0.995      0.618

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    125/149      4.71G     0.0597    0.01879    0.03323         26        416: 100% 1/1 [00:00<00:00,  9.34it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.74it/s]
                   all          3          3      0.459          1      0.995      0.618

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    126/149      4.71G    0.06236     0.0275    0.03328         34        416: 100% 1/1 [00:00<00:00,  8.74it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.47it/s]
                   all          3          3      0.459          1      0.995      0.618

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    127/149      4.71G    0.03327    0.01973    0.01839         27        416: 100% 1/1 [00:00<00:00,  7.63it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.65it/s]
                   all          3          3      0.458          1      0.995      0.566

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    128/149      4.71G    0.03361    0.02019    0.02031         29        416: 100% 1/1 [00:00<00:00,  7.83it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.27it/s]
                   all          3          3      0.458          1      0.995      0.566

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    129/149      4.71G    0.03565    0.02309    0.02501         31        416: 100% 1/1 [00:00<00:00,  8.83it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.40it/s]
                   all          3          3      0.458          1      0.995      0.566

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    130/149      4.71G    0.03527    0.01901    0.02206         26        416: 100% 1/1 [00:00<00:00,  9.62it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.61it/s]
                   all          3          3      0.458          1      0.995      0.566

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    131/149      4.71G    0.03343    0.02613     0.0212         36        416: 100% 1/1 [00:00<00:00,  7.61it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 23.36it/s]
                   all          3          3      0.402          1      0.995      0.542

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    132/149      4.71G    0.04304    0.02055    0.02943         29        416: 100% 1/1 [00:00<00:00,  9.20it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.00it/s]
                   all          3          3      0.402          1      0.995      0.542

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    133/149      4.71G    0.03176    0.01714    0.02058         23        416: 100% 1/1 [00:00<00:00,  9.65it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 25.27it/s]
                   all          3          3      0.402          1      0.995      0.542

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    134/149      4.71G     0.0637    0.02397    0.03303         33        416: 100% 1/1 [00:00<00:00,  8.89it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 24.88it/s]
                   all          3          3      0.402          1      0.995      0.542

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    135/149      4.71G    0.03491    0.02443    0.01922         33        416: 100% 1/1 [00:00<00:00,  8.23it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.45it/s]
                   all          3          3       0.39          1      0.995       0.58

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    136/149      4.71G    0.03332    0.01824     0.0184         26        416: 100% 1/1 [00:00<00:00,  9.68it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.14it/s]
                   all          3          3       0.39          1      0.995       0.58

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    137/149      4.71G     0.0685    0.02451    0.03197         33        416: 100% 1/1 [00:00<00:00,  8.32it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.31it/s]
                   all          3          3       0.39          1      0.995       0.58

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    138/149      4.71G    0.08478    0.02007    0.02759         27        416: 100% 1/1 [00:00<00:00,  9.20it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.45it/s]
                   all          3          3       0.39          1      0.995       0.58

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    139/149      4.71G    0.03447    0.01916    0.02121         25        416: 100% 1/1 [00:00<00:00,  7.57it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.28it/s]
                   all          3          3      0.379          1      0.995      0.528

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    140/149      4.71G    0.05827    0.02335    0.03419         33        416: 100% 1/1 [00:00<00:00,  8.00it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.64it/s]
                   all          3          3      0.379          1      0.995      0.528

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    141/149      4.71G    0.03634    0.02052     0.0198         30        416: 100% 1/1 [00:00<00:00,  9.60it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.59it/s]
                   all          3          3      0.379          1      0.995      0.528

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    142/149      4.71G    0.03315    0.01778    0.02149         25        416: 100% 1/1 [00:00<00:00,  9.25it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.86it/s]
                   all          3          3      0.379          1      0.995      0.528

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    143/149      4.71G    0.07553    0.02316    0.03354         32        416: 100% 1/1 [00:00<00:00,  6.42it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 28.53it/s]
                   all          3          3      0.376          1      0.995      0.543

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    144/149      4.71G    0.05824    0.01573    0.03365         24        416: 100% 1/1 [00:00<00:00,  9.32it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.84it/s]
                   all          3          3      0.376          1      0.995      0.543

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    145/149      4.71G    0.03234      0.019     0.0182         24        416: 100% 1/1 [00:00<00:00,  7.28it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.56it/s]
                   all          3          3      0.376          1      0.995      0.543

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    146/149      4.71G     0.0345    0.02095     0.0179         30        416: 100% 1/1 [00:00<00:00,  9.32it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.08it/s]
                   all          3          3      0.376          1      0.995      0.543

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    147/149      4.71G    0.07075    0.02362    0.02364         34        416: 100% 1/1 [00:00<00:00,  6.29it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 26.82it/s]
                   all          3          3      0.408          1      0.995      0.513

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    148/149      4.71G    0.03596    0.01923    0.02126         25        416: 100% 1/1 [00:00<00:00,  9.42it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 20.18it/s]
                   all          3          3      0.408          1      0.995      0.513

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
    149/149      4.71G    0.03674    0.02366    0.01994         31        416: 100% 1/1 [00:00<00:00,  8.02it/s]
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 27.57it/s]
                   all          3          3      0.408          1      0.995      0.513

150 epochs completed in 0.025 hours.
Optimizer stripped from runs/train/exp2/weights/last.pt, 14.3MB
Optimizer stripped from runs/train/exp2/weights/best.pt, 14.3MB

Validating runs/train/exp2/weights/best.pt...
Fusing layers... 
Model summary: 213 layers, 7018216 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R     mAP@.5 mAP@.5:.95: 100% 1/1 [00:00<00:00, 11.52it/s]
                   all          3          3      0.458          1      0.995      0.618
              cachorro          3          2      0.466          1      0.995      0.439
                raposa          3          1      0.451          1      0.995      0.796
Results saved to runs/train/exp2
  ```
</details>

### Evidências do treinamento

  Evidências em anexo


## Roboflow
https://app.roboflow.com/datasetimagens/dataset_imagens/overview

  
