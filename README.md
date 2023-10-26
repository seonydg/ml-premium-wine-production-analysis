# 와인 품질 일정한 등급화 &amp; 프리미엄 등급 생산 증대를 위한 중요인자 분석




# CH 01 - 와인 맛에 변화를 주는 주요 인자는 무엇인가?



# 문제 정의
동일 등급의 와인 맛에 대한 변화로 인해 고객 클레임이 발생하여 해결하려 한다.
와인은 1-9등급까지의 등급이 있고 7등급부터 프리미엄 와인으로 고가에 판매가 되는데, 공정에서 최대한 많은 프리미엄 와인의 생산을 증대시키려고 한다.
품질 등급에 영향을 끼치는 공졍을 확인하여 일정한 맛의 와인을 프리미엄 등급으로 생산을 증대하고자 한다.



# 데이터 확인

데이터 컬럼 상세
![](https://velog.velcdn.com/images/seonydg/post/858a3944-2d86-472d-aaec-de4a37f55877/image.png)

레드와인
![](https://velog.velcdn.com/images/seonydg/post/620c91c9-a712-4508-84fd-8f048c395b94/image.png)

화이트 와인
![](https://velog.velcdn.com/images/seonydg/post/a272dadd-9b5c-4ede-b673-e32dcbf566cf/image.png)



# EDA & 전처리

## 기본 정보

데이터의 가장 낮은 등급은 3이고 높은 등급은 9인데, 최소값과 최대값을 살펴보면 낮은 등급은 수치들이 대부분 낮은 분포를 가지고 있고 높은 등급은 높은 수치들의 분포를 가지고 있다.
평균값과 최대값의 차이는 눈에 띄게 차이가 나는 것은 아니지만,
좋은 등급은 첨가물 수치의 차이가 나는 것으로 예상된다.

![](https://velog.velcdn.com/images/seonydg/post/2e59a30a-cc75-4529-9309-2b56c872d5d8/image.png)

모두 수치형 데이터들이다.

![](https://velog.velcdn.com/images/seonydg/post/70333ff5-62ca-4870-bbf4-d6f36481a34d/image.png)

결측치는 없다.

![](https://velog.velcdn.com/images/seonydg/post/6c0e99ab-5401-4f6a-b2b6-597916ea2747/image.png)


## 품질 정보 확인

품질의 정보가 세분화되어 있으나, 데이터가 세분화되어 있는 것에 비해서 데이터의 샘플이 적다. 
그래서 가장 많은 품질 등급인 6을 중심으로 6보다 낮으면 1, 6이면 2, 6보다 높으면 3으로 등급을 분류한다.

![](https://velog.velcdn.com/images/seonydg/post/009ae3ed-b7a7-4b67-b85a-3a3abf090a89/image.png)

데이터들의 컬럼들의 데이터 분포도를 살펴보자.
3개의 컬럼이 1.5 수치를 넘는 것으로 비대칭성을 지니고 있다고 보인다.

![](https://velog.velcdn.com/images/seonydg/post/a3dc49fe-e82e-44ae-8a22-f1e49a48ccca/image.png)

대부분의 수치들이 왼쪽으로 치우쳐있는 것을 볼 수 있다.
여기에서 치우침을 완화할 것인지 고민을 해봐야 한다.
하지만 높은 등급을 가려내기 위해서 치우침 자체가 그 등급을 정한다면 치우침은 일단 그냥 두기로 한다.
```
for i in range(12):
    plt.subplot(3, 4, 1+i)
    plt.grid(False)
    sns.distplot(df.iloc[:, i])
    plt.ylabel(df.columns[i])

plt.gcf().set_size_inches(25, 9)
plt.tight_layout()
plt.show()
```

![](https://velog.velcdn.com/images/seonydg/post/733ba574-01fd-4ed0-9c10-f7ff120e9f50/image.png)

그럼 quality별로 그래프를 그려보자.
등급이 높아지면, 왼쪽으로 치우쳐있는 수치들이 오른쪽으로 수치들이 움직이는 것을 볼 수 있다.
수치 자체가 등급에 영향을 끼치는 것이 아닌지 다시 한 번 예측이 된다.
추가적으로 데이터의 샘플이 적어 레드 와인과 화이트 와인의 데이터를 섞었기 때문에 나타나는 데이터 수치일 수도 있다.

![](https://velog.velcdn.com/images/seonydg/post/9b6ae53d-3e66-45d8-ac6b-964731fa292d/image.png)![](https://velog.velcdn.com/images/seonydg/post/b6f2902b-c208-4a36-8ed8-fd278d08fe6f/image.png)![](https://velog.velcdn.com/images/seonydg/post/e2a2671c-b3ce-4da8-9c9c-7a526fbdc1fe/image.png)


## PCA & scaling

scaling
- 보통은 연속형 데이터들은 표준화나 정규화를 하게 되면 모델의 성능을 향상시킬 가능성이 있다.
- PCA를 진행하기 전에 scaling처리를 해야 성능이 보통 잘 나온다.

PCA
- 차원이 증가함에 따라 모델 학습 시간이 정비례하게 증가함(현재 데이터는 샘플이 적어서 학습 시간에 영향이 없어 보인다)
- 차원이 증가함에 따라서 각 결정 공간에 포함되는 샘플 수가 적어져, 과적합으로 인해 성능 저하 발생.

PCA를 2개를 적용시켜 그 값의 평균을 기준으로 다시 등급을 분리하여 진행하도록 한다.

이제 StandardScaler(표준화) scaling과 PCA(데이터 컬럼들의 차원 축소)를 적용해보자.

참조 : [변수 분포 문제](https://velog.io/@seonydg/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC-%EB%B3%80%EC%88%98-%EB%B6%84%ED%8F%AC-%EB%AC%B8%EC%A0%9C-%ED%8A%B9%EC%A7%95-%EA%B0%84-%EC%83%81%EA%B4%80%EC%84%B1-%EC%A0%9C%EA%B1%B0)

PCA를 적용하였을 때, 축소한 각 차원의 설명력(ratio)과 설정한 설명력 이상을 적용할 때 선택할 차원 수 반환 함수를 작성해보자.
```
# PCA 적용
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def pca_feature_var(data, ratio): # ratio : 축소한 각 차원의 설명력(모든 차원의 설명력 합 = 1), 보통 0.9 이상의 높은 설명력을 가지는 차원들 선택
    roop_idx = data['quality'].unique() # 품질 등급

    fig, ax = plt.subplots(len(roop_idx), 1, figsize=(20, 12))

    for i, x in enumerate(roop_idx): # 품질 등급별 PCA 적용
        df1 = data[data['quality']==x]
        X = df1.drop('quality', axis=1)

        # scaling
        scaler = StandardScaler()

        # PCA 적용
        pca = PCA()

        # pipeline : scaler와 pca 적용을 묶어서 사용 가능하도록
        pipeline = make_pipeline(scaler, pca)

        # 학습
        pipeline.fit(X)
        features = range(pca.n_components_)

        feature_df = pd.DataFrame(data=features, columns=['pc_feature'])
        variance_df = pd.DataFrame(data=pca.explained_variance_ratio_, columns=['variance'])
        pc_feature_df = pd.concat([feature_df, variance_df], axis=1)

        cumsum = np.cumsum(pca.explained_variance_ratio_)
        d = np.argmax(cumsum >= ratio) + 1 # argmax 가장 높은 값을 가지는 값의 인덱스 반환, ratio(설명력)보다 큰 값을 가지는 차원 수 선택, +1(인덱스는 0부터 시작)
        singular_vector = pd.DataFrame(pca.components_.T, index=X.columns)

        print('quality :', x, '/ 선택할 차원 수 :', d, '변수 설명력 :', cumsum[d-1])
        print(singular_vector)
        print('-'*40)

        sns.barplot(ax=ax[i], x='pc_feature', y='variance', data=pc_feature_df)
        plt.xlabel('PCA feature')
        plt.ylabel('variance')
    plt.show()
```

x축은 차원을 축소시켰을 때의 차원, y축은 차원별 설명력.

![](https://velog.velcdn.com/images/seonydg/post/b6553cd3-9d43-4c53-90c5-07e673880720/image.png)

보통 특징(컬럼)들의 상관성이 높을 때, 그 상관성을 없애기 위해서 사용할 때 PCA의 설명력 0.9 이상의 차원들을 선택한다.
하지만 기존의 컬럼들을 사용하면서 PCA 차원을 기존의 데이터에 추가하기 위해 가장 높은 설명력을 갖는 2개의 차원만 선택하도록 한다.
PCA(n_components = 2) 적용

여기에서 고민해 봐야할 것은, 데이터 샘플에 비해서 특징(컬럼)이 많은 편에 속하기에 특징을 늘리는 형태가 되어 상태 공간의 크기가 커짐에 따라서 모델의 성능을 이끌어 낼 수 있을지를 생각해야 한다.
차원 축소를 진행했을 때 축소한 데이터만 사용한다면(ex. 설명력 0.9이상을 선택), 데이터를 설명할 수 있는 설명력이 기존보다 0.1부족한 데이터 설명력을 가지게 된다. 그래서 기존 데이터의 정보 손실과 특징(컬럼)이 너무 많아 상태 공간이 커짐으로 인해 모델의 성능을 이끌어낼 수 있는지 여부는 실제로 각각의 케이스를 모두 진행해 봐야 한다.(정보 손실 VS 상태 공간)

하지만 이번 진행은 데이터 중에서 어느 특징(컬럼)이 PCA를 진행한 2개의 차원 평균 값에 영향을 많이 미치는지 확인하기 위한 것이다.
즉, PCA의 분포를 기준으로 평균값에 가까운지 멀어지는지를 가지고 진행한다.

축소된 차원의 그래프를 등급별로 확인해보자.
중심에서 멀어질수록 공정 변수의 평균과 떨어진 데이터인데, 중심에서 멀어진 데이터들이 등급에 맞는 맛과 다른 품질을 가지는 데이터라는 것을 유추해볼 수 있다.
```
 def pca_plot(df,y) :

       x=df.drop(['quality'], axis=1).reset_index(drop=True)
       y=df[y].reset_index(drop=True)

       X_ = StandardScaler().fit_transform(x)

       pca = PCA(n_components=2)
       pc = pca.fit_transform(X_)

       pc_df=pd.DataFrame(pc,columns=['PC1','PC2']).reset_index(drop=True)
       pc_df=pd.concat([pc_df,y],axis=1)

       plt.rcParams['figure.figsize'] = [10, 10]
       sns.scatterplot(data=pc_df,x='PC1',y='PC2',hue=y, legend='brief', s=100, linewidth=0)
       
pca_plot(df,'quality')
```

![](https://velog.velcdn.com/images/seonydg/post/3a8ffcef-dc2a-42dd-9d90-1652b8d9ecd8/image.png)



## PCA 차원을 통한 등급화 및 시각화
이제 기존의 데이터에 PCA 데이터를 추가시킬 것인데, 품질 등급별로 진행을 할 것이다.
중심과의 거리를 통해 한번 더 등급을 부여할 것이다.
중심과 가까울수록 공정에 맞는 데이터로써 A, B, C로 나누는 작업을 각 등급 1, 2, 3등급별로 진행한다.
그 기준은 위의 PCA를 통한 차원 축소를 한 2가지 특징을 가지고 기준을 나눌 것이다.


### 품질 등급 : 1
품질 등급이 1에 해당하는 데이터만 불러와서 차원 축소한 값을 추가하여 진행한다.
```
# 품질(Quality) 등급을 부여하기 위해, PC1값과 PC2값을 기존 데이터에 concat

df1=df[(df['quality']==1)] # 품질 1
X=df1.drop(['quality'], axis=1)
X_ = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
pc = pca.fit_transform(X_)

pc_df=pd.DataFrame(pc,columns=['PC1','PC2'])

df1_concat = pd.concat([df1.reset_index(drop=True), pc_df], axis=1)
```

그리고 'grade' 컬럼을 추가하여 축소한 차원 2개를 가지고 등급을 부여한다.
'grade'를 부여할 때, 중심에서 2미만인 데이터는 A, 4미만은 B, 그 외에는 C로 분리를 하였다.
품질의 등급을 조금 더 좁게 잡아서 진행하려면 2가 아닌, 1.5나 그 이하의 수를 적용하면 되겠다.
```
#  np.where 활용하여 등급 나누기
df1_concat['grade'] = np.where( (df1_concat['PC1']>-2) & (df1_concat['PC1']<2) & (df1_concat['PC2']>-2) & (df1_concat['PC2']<2), 'A', 
                               np.where((df1_concat['PC1']>-4) & (df1_concat['PC1']<4) & (df1_concat['PC2']>-4) & (df1_concat['PC2']<4), 'B', 'C') )
```

이제 나눈 등급을 기준으로 산점도 그래프로 시각화를 해보면 아래와 같다.
```
sns.scatterplot(data=df1_concat,x='PC1',y='PC2', s=50, linewidth=0, hue='grade');

# ▶ A grade
plt.vlines(-2, ymin=-2, ymax=2, color='r', linewidth=2);
plt.vlines(2, ymin=-2, ymax=2, color='r', linewidth=2);

plt.hlines(-2, xmin=-2, xmax=2, color='r', linewidth=2);
plt.hlines(2, xmin=-2, xmax=2, color='r', linewidth=2);

# ▶ B grade
plt.vlines(-4, ymin=-4, ymax=4, color='g', linewidth=2);
plt.vlines(4, ymin=-4, ymax=4, color='g', linewidth=2);

plt.hlines(-4, xmin=-4, xmax=4, color='g', linewidth=2);
plt.hlines(4, xmin=-4, xmax=4, color='g', linewidth=2);

plt.gcf().set_size_inches(10, 10)
```

![](https://velog.velcdn.com/images/seonydg/post/c0885165-43c6-490e-9356-ca50ed7a89fe/image.png)


축소한 차원으로 등급을 나눈 것을 가지고 기존의 데이터의 컬럼과 비교하여 어떻게 차이가 나는지 확인해보자.
아래와 같이 1등급의 와인이라도, C등급은 모든 공정변수 기준으로 평균값(center) 값에서 멀어지는 경향이 있음을 알 수 있다.
```
fig, axes = plt.subplots(4, 1)
sns.scatterplot(x=df1_concat.index, y=df1_concat['fixed acidity'], hue = df1_concat['grade'], ax=axes[0]);
sns.scatterplot(x=df1_concat.index, y=df1_concat['volatile acidity'], hue = df1_concat['grade'], ax=axes[1]);
sns.scatterplot(x=df1_concat.index, y=df1_concat['citric acid'], hue = df1_concat['grade'], ax=axes[2]);
sns.scatterplot(x=df1_concat.index, y=df1_concat['residual sugar'], hue = df1_concat['grade'], ax=axes[3]);
plt.gcf().set_size_inches(25, 15)
```

![](https://velog.velcdn.com/images/seonydg/post/a747b5fe-e179-49de-ac29-1ae72a105a15/image.png)


### 품질 등급 : 2
품질 등급이 2에 해당하는 데이터만 불러와서 차원 축소한 값을 추가하여 진행한다.
방법은 품질 등급 1과 똑같이 진행한다.

![](https://velog.velcdn.com/images/seonydg/post/bf12637c-c6b7-40f5-a7d3-4daf4f003ef5/image.png)![](https://velog.velcdn.com/images/seonydg/post/f058096d-2427-446a-9a49-8fdb82cdc096/image.png)


### 품질 등급 : 3
품질 등급이 3에 해당하는 데이터만 불러와서 차원 축소한 값을 추가하여 진행한다.

![](https://velog.velcdn.com/images/seonydg/post/19e7d0c1-2a91-4ae2-950f-45b516ce40cc/image.png)![](https://velog.velcdn.com/images/seonydg/post/ab37974a-02e3-48d0-8584-9a25f4a17f7c/image.png)

해당 데이터는 타겟 데이터의 기준치가 없기 때문에, 2차원으로 축소한 데이터의 평균값을 기준으로 다른 특징(컬럼)들의 분포를 타겟 데이터로 설정하였다.
등급별로 기존의 데이터의 컬럼과 비교하면 대체로 특징별로 A등급은 평균에 모여있고 B, C는 상대적으로 평균치에서 떨어져있는 데이터들이 조금 더 많다는 것을 확인할 수 있다.
앞쪽의 데이터들이 차이나는 이유는 레드 와인과 화이트 와인을 합쳤기에 차이가 나는 부분일 것이라 짐작이 된다.

그러면 어느 특징(컬럼)들이 와인의 품질에 영향을 많이 끼치는지 확인해보자.


# 모델링
어느 특징(컬럼)이 'grade'를 예측하는데 중요한 특징으로 작용하는지 확인하기 위해 학습 및 예측을 진행해보자.
모델은 트리 게열의 앙상블 모델인 RandomForestClassifier 사용할 것이다.
품질 등급별로 등급을 나누는데 있어 가장 크게 작용하는 특징이 있는지 살펴보자.

필요한 모듈들을 받는다.
```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import ParameterGrid
from sklearn.feature_selection import *
```

그리고 MAE를 계산하기 위해서 라벨 데이터는 숫자형으로 바꿔주는 함수를 먼저 간단히 만든다.
```
def get_grade(x):
    if x == 'A':
        x = 1
    elif x == 'B':
        x = 2
    elif x == 'C':
        x = 3
    
    return x
```

## 품질 등급 : 1
먼저 데이터를 나눈다.
```
X=df1_concat.drop(['quality', 'PC1', 'PC2', 'grade', 'n_grade'], axis=1)
Y=df1_concat['n_grade']

train_x, test_x, train_y, test_y = train_test_split(X, Y, stratify=Y)

train_x.shape, train_y.shape, test_x.shape, test_y.shape
```

그리고 하이퍼 파라미터를 설정하고
```
param_grid = ParameterGrid({
                            'n_estimators':[200, 300, 400, 500, 600],
                            'max_depth':[2, 5, 10, 15, 20],
                            'random_state':[29, 1000]
})
```

학습을 진행한다.
```
best_score = 1e9

for i, p in enumerate(param_grid):
    model = RandomForestClassifier(**p).fit(train_x, train_y)
    pred = model.predict(test_x)
    score = MAE(test_y, pred)

    if score < best_score:
        best_score = score
        best_param = p
```

score 점수가 가장 낮은 best score 성능을 내는 하이퍼 파라미터를 확인한다.
```
print(f'best_param : {best_param}')

결과:
best_param : {'max_depth': 20, 'n_estimators': 200, 'random_state': 1000}
```

재학습을 진행한다.
```
model = RandomForestClassifier(**best_param).fit(train_x, train_y)

train_pred = model.predict(train_x)
test_pred = model.predict(test_x)
```

학습을 진행한 classification report를 확인해보자.
3등급의 재현율이 다른 평가 지표들보다 낮은 것을 볼 때, C등급(3)으로 매긴 수치들에서 생산된 와인의 맛이 일정하지 않을 것이라고 예측이 된다.

![](https://velog.velcdn.com/images/seonydg/post/64648a3a-0843-4911-9a41-6a26b6ad0d5a/image.png)

마지막으로 등급을 나누는데 있어 가장 크게 작용하는 특징(컬럼)을 확인해보자.
```
ftr_importances_values = model.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = train_x.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Feature Importances')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()
```

![](https://velog.velcdn.com/images/seonydg/post/402ec866-90c9-4f92-b73c-75ff8b415300/image.png)


## 품질 등급 : 2
**품질 등급 : 1**을 확인할 때와 똑같이 진행하면 된다.

![](https://velog.velcdn.com/images/seonydg/post/d5966bee-e743-4841-8944-7aa833e106b8/image.png)
![](https://velog.velcdn.com/images/seonydg/post/6fb43e87-323b-4f64-ae1d-749f51e3afda/image.png)


## 품질 등급 : 3
**품질 등급 : 1**을 확인할 때와 똑같이 진행하면 된다.

![](https://velog.velcdn.com/images/seonydg/post/bf23b64a-9620-439b-8fb3-7fb541df0393/image.png)
![](https://velog.velcdn.com/images/seonydg/post/23d4acb5-8f13-4bcb-a256-0afe3432e172/image.png)


## 특징
품질 등급이 결정될 때, 공통적으로 가장 크게 영향을 미치는 특징은 아래와 같다.
맛의 일정함을 유지하기 위해서는 아래의 특징들을 공정 중에 특히 신경써야하는 부분이다.

- density : 밀도
- total sulfur dioxide : 총 이산화황
- chlorides : 염화물
- alcohol : 도수


# 기대효과
일정한 맛으로 인해 고객 클레임을 줄이고 회사 이미지를 긍정적으로 변화시킬 수 있다.




# CH 02



# 문제 정의
와인은 1-9등급까지의 등급이 있고 7등급부터 프리미엄 와인으로 고가에 판매가 되는데, 공정에서 최대한 많은 프리미엄 와인의 생산을 증대시키려고 한다.
품질 등급에 영향을 끼치는 공졍을 확인하여 일정한 맛의 와인을 프리미엄 등급으로 생산을 증대하고자 한다.



# EDA & 전처리

품질 주요 인자들을 탐색하고 공정 변수들을 확인하여 프리미엄 와인 생산을 증대시키는 방향으로 탐색을 한다.
품질 등급을 프리미엄과 일반으로 나누고, 
프리미엄과 일반을 구분하는 변수를 파악하기 위해 회귀 분석 모델과 분류 분석 모델을 설정하여 진행하도록 한다.


## 기본 정보 탐색
데이터의 가장 낮은 등급은 3이고 높은 등급은 9인데, 최소값과 최대값을 살펴보면 낮은 등급은 수치들이 대부분 낮은 분포를 가지고 있고 높은 등급은 높은 수치들의 분포를 가지고 있다.
최소값과 최대값, 평균값을 놓고 보았을 때, 특별히 아웃라이어로 정의할 만한 눈에 띄는 데이터는 없어 보인다.
평균값과 최대값의 차이는 눈에 띄게 차이가 나는 것은 아니지만,
좋은 등급은 첨가물 수치의 차이가 나는 것으로 예상된다.

품질의 프리미엄 등급과 아닌 등급으로 등급을 나눠서 진행하도록 한다.
프리미엄 등급(7-9등급)이면 1, 아니면(1-6등급) 0으로 설정.
프리미엄 등급이 약 20%로 클래스 불균형의 문제가 약하게 존재한다.
```
df['target'] = np.where(df['quality']>6, 1, 0)
```

![](https://velog.velcdn.com/images/seonydg/post/65034bb5-e2c5-4cc3-82c6-aadd1a1fe100/image.png)![](https://velog.velcdn.com/images/seonydg/post/a71b4953-707e-418c-b35b-9acb0af18c17/image.png)

### 클래스 불균형 확인
현재 프로젝트는 프리미엄 등급의 생산량을 늘리기 위한 주요 인자를 찾기 위한 것이 목적이기에,모델의 성능을 높이기 위한 것이 목적이 아니다.
하지만 성능을 높일수록 등급에 영향을 끼치는 주요 특징(컬럼, 인자)을 조금 더 잘 판별할 수 있을 것이라 보고 불균형을 완화하려 한다.

k-최근접 이웃 모델(KNeighborsClassifier)은 클래스 불균형에 민감한 모델로, 불균형이 존재하면 recall 수치가 낮게 측정이 되는 것을 이용한다.
n_neighbors 인자를 5에서 11까지 확인했을 시 약 20-37% 정도로 재현율이 반환되는 것으로 보아서 불균형이 존재한다.
그래서 SMOTE를 사용하여 2:1 비율까지 업샘플링을 하고, 기존과 업샘플링을 한 데이터를 모델링 시 비교해보자.

먼저 데이터를 나눈다.
```
from sklearn.model_selection import train_test_split

X = df.drop(['quality', 'target'], axis=1)
Y = df['target']

train_x, test_x, train_y, test_y = train_test_split(X, Y, stratify=Y, random_state=29)
```

불균형을 확인하는 작업을 위해 KNeighborsClassifier 모델을 활용한다.
```
# kNN을 사용한 클래스 불균형 테스트
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import *

kNN_model = KNN(n_neighbors = 5).fit(train_x, train_y)

pred_Y = kNN_model.predict(test_x)
print('recall_score: ', recall_score(test_y, pred_Y))
print('accuracy_score: ', accuracy_score(test_y, pred_Y))

결과:
recall_score:  0.3730407523510972
accuracy_score:  0.8073846153846154
```

그리고 업샘플링을 진행하기 위해 SMOTE를 사용한다.
```
from imblearn.over_sampling import SMOTE

# SMOTE 인스턴스 생성 / 인자 sampling_strategy : 업샘플링 비율 조절 인자
oversampling_instance = SMOTE(k_neighbors = 3, sampling_strategy = {1:int(train_y.value_counts().iloc[0] / 2), # 기존의 0 클래스의 크기에서 1/2 수 만큼 생성
                                                                    0:train_y.value_counts().iloc[0]}) # 기존의 0 인 클래스 수와 동일하도록

# 오버샘플링 적용
o_Train_X, o_Train_Y = oversampling_instance.fit_resample(train_x, train_y)

# ndarray 형태가 되므로 다시 DataFrame과 Series로 변환 (남은 전처리가 없다면 하지 않아도 무방)
o_Train_X = pd.DataFrame(o_Train_X, columns = X.columns)
o_Train_Y = pd.Series(o_Train_Y)
```

이제 비율을 보면 2:1 비율로 업샘플링이 되었다는 것을 확인할 수 있다.
업샘플링이 진행되었다고 해서 반드시 모델의 성능의 향상을 가져오지는 않기에,
업샘플링을 진행한 데이터와 아닌 데이터를 모델 성능 비교시 같이 본다.
중요한 점은 평가 데이터(test data)는 손을 대면 안 되기에, 미리 데이터를 나누어 놓았다.

![](https://velog.velcdn.com/images/seonydg/post/1f604923-cf1b-454a-822d-4b83a6bb8719/image.png)


## 품질 정보 탐색
특징(컬럼)들의 상관관계를 확인하기 위해 먼저 히트맵으로 확인해보자.

![](https://velog.velcdn.com/images/seonydg/post/56284cf9-5860-4150-84f2-53529fad991a/image.png)

각 특징별로 상관관계들의 절대값들을 더해서 특징수로 나눠서 상관계수가 가장 높게 나타나는 컬럼이 무엇인지 보자.
대표수치라고 할 수는 없지만, 모든 상관계수들의 합이니 합이 높을 수록 각 특정 컬럼들이 미치는 상관성을 대략적으로 볼 수 있다.
그리고 높을수록 프리미엄을 가려내는 중요한 변수가 될 확률이 높다.

![](https://velog.velcdn.com/images/seonydg/post/d00014d5-3106-4fd5-8cf7-772617b34a3a/image.png)

이제는 라벨(타겟) 데이터에 영향을 많이 주는 데이트 3개를 산점도 그래프로 그려보자.
'volatile acidity'와 'density'는 등급이 높을수록 수치가 낮은 경향을 보이고, 'alcohol'은 등급이 높을수록 수치가 높은 경향을 보인다.

![](https://velog.velcdn.com/images/seonydg/post/59cd7c71-05c0-4750-8fd3-0a2b2734fdef/image.png)



# 모델링 1 - 회귀 분석
라벨이 0과 1로 이루어져 있어 분류 문제지만 회귀로도 진행해서 설명력을 갖는지 확인해보자.
선형 회귀 LinearRegression 모델을 사용할 것이다.

0과 1사이 값의 라벨에 대해서 0.26 - 0.27값의 차이가 보이고, r2 score는 설명력인데 그다지 좋아보이지 않는다.
```
# LR(선형회귀) 모델 활용
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_absolute_error, r2_score

lr = LR()
lr.fit(train_x, train_y)

pred_train = lr.predict(train_x)
pred_test = lr.predict(test_x)

mae_train = mean_absolute_error(train_y, pred_train)
r2_train = r2_score(train_y, pred_train)

mae_test = mean_absolute_error(test_y, pred_test)
r2_test = r2_score(test_y, pred_test)

print(f'mae train: ', mae_train)
print('r2 train: ', r2_train)
print('-'*20)
print('mae test: ', mae_test)
print('r2 test: ', r2_test)

결과:
mae train:  0.267818619333233
r2 train:  0.19844504769653282
--------------------
mae test:  0.27467166147572714
r2 test:  0.17100151159564436
```

그리고 클래스 불균형을 조금 해소한 데이터로 다시 진행해보자.
정답과 예측값의 차이는 더 벌어졌고, r2를 보면 설명력도 줄어들어서 업샘플링한 것의 좋지 못한 결과를 볼 수 있다.
```
lr = LR()
lr.fit(o_Train_X, o_Train_Y)

pred_train = lr.predict(o_Train_X)
pred_test = lr.predict(test_x)

mae_train = mean_absolute_error(o_Train_Y, pred_train)
r2_train = r2_score(o_Train_Y, pred_train)

mae_test = mean_absolute_error(test_y, pred_test)
r2_test = r2_score(test_y, pred_test)

print(f'mae train: ', mae_train)
print('r2 train: ', r2_train)
print('-'*20)
print('mae test: ', mae_test)
print('r2 test: ', r2_test)

결과:
mae train:  0.34575993988535864
r2 train:  0.24948126270298
--------------------
mae test:  0.31956417608207155
r2 test:  0.07133318420475421
```

사실 분류 문제를 가지고 회귀분류를 진행한 것은 선형 모델을 사용하려는 것 보다는, 모델의 계수(model.coef_)를 보여주기 때문이다.
계수를 보면 'density'는 음수로 density가 낮아져야 등급이 올라가며, 'sulphates' 양수로 sulphates가 높아지면 등급이 올라간다고 유추해 볼 수 있다.

![](https://velog.velcdn.com/images/seonydg/post/ea7ca1e7-ffc9-4cfe-8339-5ab82a9872f0/image.png)

회귀 문제를 볼 때에는 다중공선성도 생각해야 한다.
다시 말해서 특징간의 상관성이 높을수록 수치가 높은 특징만 반영이 되고 다른 특징은 반영이 되지 않을 수 있거나, 여러 특징들 중 어느 특징에 의해서 결과가 도출이 되었는지 확인이 어려운 문제가 있다.
그래서 표준화 및 정규화로 완화 및 PCA를 통해서 해소하거나 상관성이 있는 특징(컬럼) 중 하나를 삭제하는 방향으로 갈 수도 있다.
여기에서는 다중공선성을 확인하고 제거하는 방향으로 진행해보자.

상관계수가 0.5 이상인 것들만 가져와서 확인하고 데이터를 다시 분할하여 진행한다.

![](https://velog.velcdn.com/images/seonydg/post/de901c76-6130-43de-b367-150f0329556e/image.png)

'residual sugar', 'free sulfur dioxide' 두개의 특징을 삭제하고 다시 진행해본다.
mae값이 조금 좋아졌지만 r2값도 떨어진 것을 볼 수 있다.
```
X = df.drop(['residual sugar', 'free sulfur dioxide', 'quality', 'target'], axis=1)
Y = df['target']

train_x, test_x, train_y, test_y = train_test_split(X, Y, stratify=Y)

lr.fit(train_x, train_y)
train_pred = lr.predict(train_x)
test_pred = lr.predict(test_x)

mae_train = mean_absolute_error(train_y, train_pred)
r2_train = r2_score(train_y, train_pred)

mae_test = mean_absolute_error(test_y, test_pred)
r2_test = r2_score(test_y, test_pred)

print(f'mae train: ', mae_train)
print('r2 train: ', r2_train)
print('-'*20)
print('mae test: ', mae_test)
print('r2 test: ', r2_test)

결과:
mae train:  0.2708489542877124
r2 train:  0.1820203764199536
--------------------
mae test:  0.27115793184807047
r2 test:  0.16415797208797311
```

달라진 점은 density의 계수가 양수로 바뀌었다.
생각해볼 것은, 음에서 양으로 바뀌었다면 특징 2가지를 지우기 전에 모델의 해석이 바뀌었을 수 있다. 그리고 원래 양수였는데 음수로 잘못 나왔었는데 이제는 제대로 나온 것일 수도, 원래 음의 계수가 맞는데 지금이 잘못 나온 것일 수도 있다.

회귀 분석으로만 본다면, 계수가 높은 순으로 density, chlorides, volatile acidity, sulphates, alcohol 5개 정도가 등급에 영향을 크게 미치는 특징으로 관리를 해야 하는 인자라고 볼 수 있다.
하지만 분류 문제를 회귀 문제로 가져와서 보는 것도 의미가 있을 수는 있으나 너무 깊게 들어가지는 않겠다.

![](https://velog.velcdn.com/images/seonydg/post/9051d887-1694-4f52-bd6f-f774cbf4ad9c/image.png)


# 모델링 2 - 분류 분석
모델은 앙상블 계열의 LGBMClassifier과 RandomForestClassifier을 가지고 진행을 해보자.
```
from lightgbm import LGBMClassifier as LGB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score, accuracy_score
```

기존과 업샘플링을 진행한 데이터를 가지고 진행한 후, 비교를 위해 다시 업샘플링을 진행한다.
데이터를 다시 나누고,
```
X = df.drop(['quality', 'target'], axis=1)
Y = df['target']

train_x, test_x, train_y, test_y = train_test_split(X, Y, stratify=Y, random_state=29)
```
다시 업샘플링을 진행한다.
```
# SMOTE 인스턴스 생성
oversampling_instance = SMOTE(k_neighbors = 3, sampling_strategy = {1:int(train_y.value_counts().iloc[0] / 2), # 기존의 -1 클래스의 크기에서 1/2 수 만큼 생성
                                                                    0:train_y.value_counts().iloc[0]}) # 기존의 -1 인 클래스 수와 동일하도록

# 오버샘플링 적용
o_Train_X, o_Train_Y = oversampling_instance.fit_resample(train_x, train_y)

# ndarray 형태가 되므로 다시 DataFrame과 Series로 변환 (남은 전처리가 없다면 하지 않아도 무방)
o_Train_X = pd.DataFrame(o_Train_X, columns = X.columns)
o_Train_Y = pd.Series(o_Train_Y)
```

![](https://velog.velcdn.com/images/seonydg/post/86784a33-4107-4e11-a926-e037d0e05f49/image.png)

하이퍼 파라미터 튜닝을 진행을 할 것인데, 
다음과 같이 인스턴스화 된 모델에 get_params() 속성을 확인하면, 조절해야 할 파라미터들과 그 default값을 보여준다.

![](https://velog.velcdn.com/images/seonydg/post/bfef9513-f03b-45bf-85d0-22f0c569b951/image.png)

```
model_parameter_dict = dict()

RFR_param_grid = ParameterGrid({
                                'max_depth':[3, 5, 10, 15],
                                'n_estimators':[100, 200, 400],
                                'n_jobs':[-1]
})
LGB_param_grid = ParameterGrid({
                                'max_depth':[3, 5, 10, 15],
                                'n_estimators':[100, 200, 400],
                                'learning_rate':[0.05, 0.1, 0.2]
})

model_parameter_dict[RFC] = RFR_param_grid
model_parameter_dict[LGB] = LGB_param_grid
```

f1 score를 평가지표로 삼아서 학습 및 예측을 하여 최적의 모델을 선택한다.
```
best_score = -1
iteration_num = 0

for m in model_parameter_dict.keys():
    for p in model_parameter_dict[m]:
        model = m(**p).fit(train_x.values, train_y.values)
        pred = model.predict(test_x.values)
        score = f1_score(test_y.values, pred)
        acc = accuracy_score(test_y.values, pred)

        if score > best_score:
            best_score = score
            best_model = m
            best_parameter = p
            accuracy = acc
            
        iteration_num += 1

        print(f'iter_num-{iteration_num}/{max_iter} => score : {score:.3f}, best score : {best_score:.3f} | acc : {accuracy}')
```

최고 점수를 뽑아낸 모델과 파라미터, 그리고 점수는 다음과 같다.
lightgbm 모델은 데이터의 최소 샘플수가 1만개는 넘어갈 때 쓰는 것이 보통 좋은 성능을 낸다고 한다.

![](https://velog.velcdn.com/images/seonydg/post/51959bdc-6416-45a0-b36f-4a9ebcc60f70/image.png)

그리고 최적의 모델과 파라미터를 가지고 재학습하여 classification_report를 확인해보자.
```
# best 모델 학습
model = best_model(**best_parameter).fit(train_x.values, train_y.values)

train_pred = model.predict(train_x.values)
test_pred = model.predict(test_x.values)
```

재현율이 조금 낮은 것을 볼 수 있다.
그런데 학습 데이터로 다시 돌려보았을 때 모두 1이 나온 것으로 봐서 과적합일 확률이 높아 보인다.
하지만 평가 데이터의 타겟값 0(일반 등급)의 값은 평가 지표가 좋은 것을 보면 약간의 과적합 정도일 것 같아 보이기도 한다.
이럴 때에는 LGBMClassifier보다는 RandomForestClassifier의 파라미터를 조금 더 세세하게 조정해서 성능을 조금 높여보는 것도 좋겠다.
LGBMClassifier의 방식이 부스팅 방식이라서 데이터의 샘플이 적으면 과적합이 될 우려가 있다.

![](https://velog.velcdn.com/images/seonydg/post/e13a172c-dc65-4443-b276-08f394fed7d1/image.png)


## 업샘플링한 데이터로 다시
업샘플링한 데이터로 f1 score로 최적의 모델을 확인한다.
```
best_score = -1
iteration_num = 0

for m in model_parameter_dict.keys():
    for p in model_parameter_dict[m]:
        model = m(**p).fit(o_Train_X.values, o_Train_Y.values)
        pred = model.predict(test_x.values)
        score = f1_score(test_y.values, pred)
        acc = accuracy_score(test_y.values, pred)

        if score > best_score:
            best_score = score
            best_model = m
            best_parameter = p
            accuracy = acc

        iteration_num += 1

        print(f'iter_num-{iteration_num}/{max_iter} => score : {score:.3f}, best score : {best_score:.3f} | acc : {accuracy}')
```

그리고 확인을 해보면 평가 지표 accuracy는 0.7정도 하락했고 f1 score는 0.4정도 올라서 별차이가 없어보인다.

![](https://velog.velcdn.com/images/seonydg/post/68c9f3d3-027c-4184-a0e3-e756cbeec7ad/image.png)

```
# best 모델 학습
model = best_model(**best_parameter).fit(o_Train_X.values, o_Train_Y.values)

train_pred = model.predict(o_Train_X.values)
test_pred = model.predict(test_x.values)
```

하지만 아래의 classification_report를 보면 재현율(recall)이 타겟값 1(프리미엄 등급)에 대한 지표가 4정도 올라간 것을 볼 수 있다.
평가 지표 전체를 놓고 보면 분류에서는 업샘플링한 데이터로 학습한 모델이 조금 더 좋은 성능을 보인다고 해석할 수 있다.
아니면 lightgbm의 모델은 데이터의 샘플수가 많을 수록 좋은 성능을 낸다고 알려져 있으니, 
업샘플링으로 인해 데이터가 많아져서 좋은 성능이 나왔다고 해석할 수도 있다.

![](https://velog.velcdn.com/images/seonydg/post/a4cdd91d-021d-43b4-907a-d0fb87ffea6e/image.png)

마지막으로 분류 모델을 통해서 타겟값을 예측할 때 중요한 변수는 다음과 같다.
```
ftr_importances_values = model.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:21]

plt.figure(figsize=(12,8))
plt.title('Feature Importances')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()
```

예상했던 것 보다 density의 중요도가 회귀 분석 때와는 다르게 낮고, 낮은 계수를 가지던 total sulfur dioxide 특징이 높아진 것을 볼 수 있다. 그리고 total sulfur dioxide와 free sulfur dioxide은 서로 상관 관계를 가지고 있었던 특징으로 적용되어 둘 다 같이 확률이 높다.

하지만 회귀 분석 때에는 r2의 설명력이 낮아서 신뢰하지 못하는 수준이라 판단이 되고,
해당 분류 분석의 지표가 더 신뢰할 수 있을 것이라 판단 된다.

![](https://velog.velcdn.com/images/seonydg/post/3874ad5a-c6ee-4a53-ad85-573464e5632a/image.png)



# 기대효과
생산 공정에서 프리미엄 등급에 영향을 끼치는 인자를 판별하여 조금 더 신중하게 관리를 함으로써 프리미엄 등급의 생산을 증대시킬 수 있다.
이제 일정한 맛을 내는 프리미엄 등급의 와인 생산을 증대함으로써 영업 이익을 높일 수 있다.
