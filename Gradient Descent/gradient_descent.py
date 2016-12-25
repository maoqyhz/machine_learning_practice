#encoding:utf-8

#随机梯度
def stochastic_gradient_descent(x,y,theta,alpha,m,max_iter):
	"""随机梯度下降法，每一次梯度下降只使用一个样本。

	:param x: 训练集种的自变量
	:param y: 训练集种的因变量
	:param theta: 待求的权值
	:param alpha: 学习速率
	:param m: 样本总数
	:param max_iter: 最大迭代次数
	"""
	deviation = 1
	iter = 0	
	flag = 0
	while True:
		for i in range(m):	#循环取训练集中的一个
			deviation = 0
			h = theta[0] * x[i][0] + theta[1] * x[i][1]
			theta[0] = theta[0] + alpha * (y[i] - h)*x[i][0] 
			theta[1] = theta[1] + alpha * (y[i] - h)*x[i][1]

			iter = iter + 1
			#计算误差
			for i in range(m):
				deviation = deviation + (y[i] - (theta[0] * x[i][0] + theta[1] * x[i][1])) ** 2
			if deviation <EPS or iter >max_iter:
				flag = 1 
				break
		if flag == 1 :
			break	
	return theta, iter

#批量梯度
def batch_gradient_descent(x,y,theta,alpha,m,max_iter):
	"""批量梯度下降法，每一次梯度下降使用训练集中的所有样本来计算误差。

	:param x: 训练集种的自变量
	:param y: 训练集种的因变量
	:param theta: 待求的权值
	:param alpha: 学习速率
	:param m: 样本总数
	:param max_iter: 最大迭代次数
	"""
	deviation = 1
	iter = 0
	while deviation > EPS and iter < max_iter:
		deviation = 0
		sigma1 = 0
		sigma2 = 0
		for i in range(m): #对训练集中的所有数据求和迭代
			h = theta[0] * x[i][0] + theta[1] * x[i][1]
			sigma1 = sigma1 +  (y[i] - h)*x[i][0] 
			sigma2 = sigma2 +  (y[i] - h)*x[i][1] 
		theta[0] = theta[0] + alpha * sigma1 /m
		theta[1] = theta[1] + alpha * sigma2 /m
		#计算误差
		for i in range(m):
			deviation = deviation + (y[i] - (theta[0] * x[i][0] + theta[1] * x[i][1])) ** 2
		iter = iter + 1
	return theta, iter


#运行 为两种算法设置不同的参数
# data and init 
matrix_x = [[2.1,1.5],[2.5,2.3],[3.3,3.9],[3.9,5.1],[2.7,2.7]]
matrix_y = [2.5,3.9,6.7,8.8,4.6]
MAX_ITER = 5000
EPS = 0.0001 
#随机梯度
theta = [2,-1]
ALPHA = 0.1

resultTheta,iters = stochastic_gradient_descent(matrix_x, matrix_y, theta, ALPHA, 5, MAX_ITER)
print 'theta=',resultTheta
print 'iters=',iters

#批量梯度
theta = [2,-1]
ALPHA = 0.1
resultTheta,iters = batch_gradient_descent(matrix_x, matrix_y, theta, ALPHA, 5, MAX_ITER)
print 'theta=',resultTheta
print 'iters=',iters
