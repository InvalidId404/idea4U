import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf


class K_MeanClustering:  # K-Means 알고리즘을 이용한 군집화 클래스, 데이터세트와 중심점의 개수(k)를 받아 중심점의 위치를 구한다
    def __init__(self, dataset, k):  # Dataset : 데이터세트, K : 군집의 중심점 개수
        self.dataset = dataset  # (x, y)를 원소로 갖는 2차원 리스트
        self.k = k  # 군집에서의 중심 개수
        self.num_points = len(self.dataset)  # 점 개수
        self.centroids = []  # 중심점의 좌표 (x, y)를 원소로 갖는 2차원 리스트
        self.centroid_values = 0  # 중심점의 값
        self.assignment_values = 0

    def cluster(self, k):  # 군집화 메서드, 중심의 좌표를 1차원 리스트로 반환
        vectors = tf.constant(self.dataset)  # 데이터세트로부터 상수 텐서 생성
        centroids = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))  # 데이터세트 중에서 무작위로 중심점 설정

        ex_vectors = tf.expand_dims(vectors, 0)  # 텐서 차원 확장(vectors와 centroids 텐서 간 뺄셈을 위해)
        ex_centroids = tf.expand_dims(centroids, 1)

        assignments = tf.argmin(  # reduce_sum 을 최소로 만드는 중심점을(D0 차원, 인자 2) 구해 assignments 에 저장한다==최단거리를 만드는 tf.square 값
            tf.reduce_sum(  # (x1-x2)^2 + (y1-y2)^2 값을 구하고 두 차원을 병합한다
                tf.square(tf.subtract(ex_vectors, ex_centroids)), 2),  # 유클리드 제곱거리
            0)
        '''
        <점-점 거리 공식(유클리드 제곱거리)을 이용하여 최단거리에 위치한 중심점을 찾는다>
        # 1. 확장된 벡터, 중심 간 좌표차의 제곱을 구한다 (square)
        # 2. 제곱의 합을 구한다 == 거리(reduce_sum) 이때 2번째 차원은 없어짐
        # 3. tf.argmin 으로 가장 적은 거리를 만드는 중심을 구해서 반환한다
        '''
        means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])),
                                          reduction_indices=[1]) for c in range(k)], 0)  # 군집 내 평균 계산

        update_centroids = tf.assign(centroids, means)  # 새로운 중심 설정

        init_op = tf.global_variables_initializer()  # 변수 초기화

        sess = tf.Session()
        sess.run(init_op)

        for step in range(100):
            _, self.centroid_values, self.assignment_values = sess.run([update_centroids, centroids, assignments])

    def draw(self):
        data = {'x': [], 'y': [], 'cluster': []}

        for i in range(len(self.assignment_values)):
            data['x'].append(self.dataset[i][0])
            data['y'].append(self.dataset[i][1])
            data['cluster'].append(self.assignment_values[i])

        df = pd.DataFrame(data)
        sns.lmplot('x', 'y', data=df, fit_reg=False, size=6, hue='cluster', legend=False)

        plt.show()


def main():
    num_vectors = 5000
    num_clusters = 4
    num_steps = 100
    vector_values = []
    for i in range(num_vectors):
      if np.random.random() > 0.5:
        vector_values.append([np.random.normal(0.5, 0.6),
                              np.random.normal(0.3, 0.9)])
      else:
        vector_values.append([np.random.normal(2.5, 0.4),
                             np.random.normal(0.8, 0.5)])

    k = K_MeanClustering(vector_values, 4)
    k.cluster(4)
    k.draw()

if __name__ == '__main__':
    main()
