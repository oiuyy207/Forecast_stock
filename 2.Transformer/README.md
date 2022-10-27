# Transformer 모델 개발(~22-09-02)  

## 디렉토리
1. **img** : 이미지 파일 모음. 단, 학습 과정에서 나온 모델 구조 그림은 transformer 디렉토리 내 각 파일에 들어있다.(단, 초기버전에는 없음)  
2. **이전코드** : 개발 과정에서 이전에 사용한 코드.  
3. **util** : 모델 개발 과정에서 여러가지 함수 모음. MI-LSTM 개발 당시 있던 함수들에서 가져옴.  
4. **model_layer** : transformer 모델 레이어들.  
5. **stock/stock_test** : 데이터 파일 모음. 코드 동작 중에 생성된다.  
6. **transformer** : 모델 결과물.  
7. **etc** : 기타 등등 사용/생성한 파일  

----------

## 그외 파일  
> M6_LJY_220630.yaml : 22/06/30기준 가상환경.  
> M6_Universe.csv : M6 Compitition에 사용되는 파일. 예측해야할 asset의 이름, 코드들이 들어있다.  

-----------

## (10월 27일) 현재 연구 중  
**time_attention**을 만들고, **siamese_enc_for_multi_input**에 적용시키기 위해 연구중  
> **multi-enc-transformer_V0.1.ipynb**에서 작업 중  
  
