# Naver_BoostCamp_NOTA

### Installation
1. 도커 파일 다운로드 후 압축 해제

```bash
git clone https://github.com/nota-github/Naver_BoostCamp_NOTA.git
```

2. 모델 환경이 정의된 도커 이미지 생성
```bash
cd np_app_segformer
# 이미지명:태그 = notadockerhub/np_app_segformer:latest
docker build -t notadockerhub/boostcamp:latest -f ./Dockerfile .
```
### Training
1. 데이터셋 준비
[ADE20K](https://drive.google.com/file/d/1cBd9z93CfI6v-fsIHqLc58fVEH2urJyx/view?usp=share_link)

2. 모델을 구동할 도커 컨테이너 생성하기
```bash
docker run --name {container_name} --shm-size={usable memory} -it --gpus all -v /{ade20k가 들어가 있는 dir}:/root/datasets notadockerhub/boostcamp:latest

# example(dataset/ADEChallengeData2016)
docker run --name segformer_challenge --shm-size=8g -it --gpus all -v /root/dataset/:/root/datasets notadockerhub/boostcamp:latest
```

3. 학습 시작

```bash
# 현재 디렉토리: /root/Naver_BoostCamp_NOTA
python train.py \
    --data_dir {ADE20K의 path} \
    --device 0,1,2,3 \ # 환경에 맞게 수정 
    --save_path {save하고자 하는 dir의 path} \ 
    --pretrain {pretrain 모델 dir의 path} # fine-tuning일 경우 기입
```

### ImageNet Training
1. ImageNet download

2. segformer 모델 import 부분 수정(필요시)
- [main.py 22th line](https://github.com/nota-github/Naver_BoostCamp_NOTA/blob/main/imagenet_pretrain/main.py#L22) 
- [main.py 256th line](https://github.com/nota-github/Naver_BoostCamp_NOTA/blob/main/imagenet_pretrain/main.py#L256)

3. training
```bash
sh dist_train.sh {사용하는 gpu 개수} --data-path {imagenet path} --output_dir {save dir path}

# example
sh dist_train.sh 4 --data-path /workspace/imagenet --output_dir result/mod_segformer/
```

### Evaluation & FLOPs, 파라미터 개수 확인
- evaluate 수행

```bash
# phase를 통해 val 또는 test set 설정
python eval.py \ # eval.py 내의 model을 정의하는 코드 수정
	--data_dir {ADE20K의 path} \
    --pretrain {pretrain 모델 dir의 path}
```

- 최종 산출물
    - val set에 대한 evaluation 결과
        - 전체 mIoU
        - 전체 Accuracy
        - label별 mIoU

- FLOPs, 파라미터 개수 확인

```bash
python util/get_flops_params.py \ # get_flops_params.py 내의 model을 정의하는 코드 수정
    --data_dir {ADE20K의 path}
```