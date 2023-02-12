# Toppan-Identification
登録した人物を、検出してトラッキングする

<br>

## **開発環境**
- Python 3.8.12
- Ubuntu 20.04
- GPU ( NVIDIA GeForce RTX 3070 Laptop GPU )

<br>

## **環境構築**
### 1. [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
1. [Pretrainedモデル ( yolox_x.pth )](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth)をdownloadしてWeights/YOLOXフォルダに格納

### 2. [ByteTrack](https://github.com/ifzhang/ByteTrack)
1. セットアップ
    ```
    mkdir Library
    git clone git@github.com:ifzhang/ByteTrack.git
    cd ByteTrack
    pip3 install -r requirements.txt
    pip3 install cython_bbox
    python3 setup.py develop
    ```

2. [Pretrainedモデル ( bytetrack_s_mot17.pth.tar )](https://drive.google.com/file/d/1uSmhXzyV1Zvb4TJJCzpsZOIcw7CCJLxj/view)をdownloadしてWeitghts/ByteTrackフォルダ下に格納

<br>

### 3. [TransRe-ID](https://github.com/damo-cv/TransReID)
1. セットアップ
    ```
    cd ../
    git clone git@github.com:damo-cv/TransReID.git
    cd TransReID
    pip3 install -r requirements.txt
    ```

2. [Pretrainedモデル ( jx_vit_base_p16_224-80ecf9dd.pth, vit_transreid_market.pth )](https://drive.google.com/drive/folders/1ZvBs7O8583yBIuhR2zcf0zm5VN_b60o_)をdownloadしてWights/TransReIDフォルダ下に格納

<br>

### 4. 残りのライブラリを整える
```
cd ../../
pip3 install -r requirements.txt
```

<br>

***

<br>

## **実行**
### 1. ローカルの場合
1. Trackerの登録  
    1. Config/registration.yaml に設定値を記入
    2. 下記実行
        ```
        python3 execute_registration_on_local.py
        ```
2. Assign-identification  
    1. Config/identification.yaml に設定値を記入
    2. Library/TransReID/configs/Market/vit_transreid.yml の3行目 (PRETRAIN_PATH)に下記を入力
        ```
        "Weights/TransReID/jx_vit_base_p16_224-80ecf9dd.pth"
        ```
    3.  Library/TransReID/configs/Market/vit_transreid.yml 11行目 (STRIDE_SIZE)に下記を入力
        ```
        [12, 12]
        ```
    4. 下記実行
        ```
        python3 execute_identification_on_local.py
        ```
<br>

### ※ [任意] 下記エラーが発生する場合は使用PCのCUDAのバージョンに合わせてtorch, torchaudio, torchvisionを指定のバージョンにアップデート [(参考URL)](https://pytorch.org/)
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
< 例 >
```
pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

### ※ [任意] 下記エラーが発生する場合はtorchのversionを1.8未満にダウングレードする  ( torchaudio, torchvisionも合わせてダウングレード ) [(参考URL)](https://pytorch.org/)
```
ImportError: cannot import name 'container_abcs' from 'torch._six' 
```
< 例 >
```
pip3 install -U torch==1.7.1 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

###  備考
execute_identification_on_localにおいて、同一フレーム内で同一人物の名前が割り当てられないようになっている。割り当てても良い場合は以下のcommit番号をpullして実行する
```
710f21ad12cd41bd1f85fb686f92425796972a37
```