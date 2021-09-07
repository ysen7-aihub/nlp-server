## NLP SERVER

```
test
∟ models
  ∟ 7emotions_model.pt
  ∟ 7emotions_model_state_dict.pt
  ∟ 7emotions_all.tar
∟ app.py
```



### library install errors

#### [목표는 nlp.py 파일이 실행되게 하는 것입니다.]

### 1.
- 현재 repository를 zip으로 다운로드해주세요. (test 폴더)

### 2.
- 구글 공유드라이브에서 현우님이 올려주신 모델zip파일을 다운로드 받아주세요.
- test폴더 안에 아래 그림과 같이 __models__ 라는 폴더를 만들고 그 안에 모델파일 3개를 넣어주세요.
![image](https://user-images.githubusercontent.com/76643037/132103385-2507fdb7-cab2-4b52-a39c-e2060b4381b5.png)

### 3.
- test 폴더 안에 각자 파이썬 가상환경을 만들고, pip3 install로 라이브러리들을 설치해주세요.
- 설치할 라이브러리
  - mxnet
  - gluonnlp
  - pandas
  - tqdm
  - sentencepiece
  - transformers
  - torch
  - git+https://git@github.com/SKTBrain/KoBERT.git@master

이 때 문제는, 라이브러리 간의 버전때문에 설치오류가 날 수 있습니다.
버전을 다르게 바꾸어가면서 모든 라이브러리들이 설치될 수 있게 해주세요(ex. transformer==2.5.1)

### 4.
- requirements.txt에는 제가 설치해본 라이브러리들입니다.
- 사용된 모든 라이브러리들을 설치 성공했지만, nlp.py 파일을 실행시켰을 때 다음과 같은 에러가 뜹니다.
- (transformer.modeling_bert 모듈이 없다)

![image](https://user-images.githubusercontent.com/76643037/132103604-0b8dd52e-6a3f-4e4a-ad18-5854c91ae737.png)

### 5.
- 이후 rustc를 설치

### 6.
같은 에러 참고 사이트
https://github.com/pydata/bottleneck/issues/281

### 7. 
0906 현재상태
- nlp.py를 실행시키기 위해서는, transformers==3.0.2를 설치해야 하는 것으로 생각중.

- rustc (nightly version)을 설치한 상태
  ##### 1. transformers 2점대 버전 또는 4점대 버전 설치했을 경우
  - rustc 설치하기 전에는 모든 버전의 transformers가 설치되지 않았지만, 설치 이후 3점대 버전을 제외한 모든 버전은 일단 설치가 됨
  - 하지만, nlp.py를 실행하면 다음과 같은 에러발생
  ![image](https://user-images.githubusercontent.com/76643037/132164930-0ffea7f4-f61f-4e6e-9503-6f7defb97876.png)
  - BertEncoder has no attributes 'output_hidden_states'

  ##### 2. transformers 3점대 버전을 설치했을 경우
  - ERROR: Could not build wheels for tokenizers which use PEP 517 and cannot be installed directly
  ![image](https://user-images.githubusercontent.com/76643037/132165115-f550912a-ee91-4413-bfe1-8d5db33d1423.png)

  ##### 3. PEP 517 error
   - python 버전의 문제인가 싶어, 3.7과 3.9에서 모두 시도했지만 실패
  
### 8.
- 현재 개발환경은 Windows
- 이후 서버를 생성하고 Ubuntu 환경에서 실행시켜야함
  - windows에서 성공하고 모든 requirements를 그대로 하면 ubuntu에서도 성공할까?
  - rustc 설치같은 경우는 requirements.txt에 어떻게 적어두어야 할까? (windows같은 경우 터미널에서 설치한 것이 아니고 인터넷에 접속 후 설치)
