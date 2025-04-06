# Deep Learning Based Damaged Car Segmentation Web Service.

이 프로젝트는 한국 Ai Hub에서 구축한 [차량 파손 이미지 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=581)를 활용하여, 
딥러닝 기반으로 자동차의 파손 부위를 세그멘테이션(Segmentation)하는 웹 서비스의 Ai Repository 입니다.

---

Web Service URL : https://www.crushedmycar.site/
---

## Preview
|Input|Output|
|------|---|
|![Screenshot from 2025-04-07 02-05-51](https://github.com/user-attachments/assets/98b4b356-c471-4508-b656-6bbb50c27879)|![Screenshot from 2025-04-07 02-05-57](https://github.com/user-attachments/assets/0654a08e-2ad1-4980-87f1-32e268a72188)|

---

## Intro
- 차량 파손 이미지를 입력받아, 파손 부분을 자동으로 인식하고 세그멘테이션하는 기능을 제공합니다.
- 웹 서비스 형태로 구현되어, 사용자들이 쉽게 업로드하고 결과를 확인할 수 있습니다.

---

## Metrics
- **Accuracy:** 92%
- **Precision:** 82%
- **Recall:** 72%

---

## Model Composition
- **UNet** 구조를 기반으로, Dataset 의 4개의 Label(스크래치, 찌그러짐, 파손, 이격)을 각각 세분화하기 위해 **4개의 UNet** 모델을 별도로 학습시켰습니다.
- 각 모델은 해당 라벨에 최적화된 학습을 진행하여, 종합적인 성능을 향상시켰습니다.

---

## Reference.
- [한국 Ai Hub – 차량 파손 이미지 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=581)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
