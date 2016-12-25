%加载数据和数据预处理
data = load ('data1.txt');
fprintf('Running Gradient Descent ...\n');

X = data(:, 1); 	%取X集合
y = data(:, 2);		%取y集合

%画出数据的散点图
figure;
plot(X, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s'); 
xlabel('Population of City in 10,000s'); 
hold on; 

%设置学习参数
X = [ones(length(data),1),data(:,1)]; 	% y = theta0*x0 + theta1*x1 默认x0为1
theta = zeros(2,1);	%theta初始值为0
alpha = 0.01;
max_iter = 5000;

m = length(y);	%数据组数
J_history = zeros(max_iter,1);	%初始化迭代误差变量
iter = 1;
for iter = 1:max_iter
	%每迭代100次画一条曲线
    if  mod(iter,100) == 0
        plot(X(:,2), X*theta, 'g')
    end

	theta = theta - alpha / m * X' * (X * theta - y);	%梯度下降
    J_history(iter) = sum((X * theta-y) .^ 2) / (2*m);	%记录每次 迭代后的全局误差
    fprintf('iter:%d ------ Error:%f\n',iter,J_history(iter));
end

%输出最终的theta值和最终的拟合曲线
disp('Theta found by gradient descent:');
disp(theta);
plot(X(:,2), X*theta, 'k')
legend('Training data', 'Linear regression')
hold off 
 