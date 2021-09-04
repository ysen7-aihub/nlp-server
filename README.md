## library install errors

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
- 한 번씩만 해주세요. 부탁드립니다ㅠㅠ
