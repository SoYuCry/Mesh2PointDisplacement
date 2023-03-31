% 三角形的三个顶点
a = [0, 1];
b = [0,0];
c = [1, 0];
x = [a(1),b(1),c(1),a(1)]
y = [a(2),b(2),c(2),a(2)]

a_new = [0.9,0];
b_new = [0,0];
c_new = [0.2,0.8];
x_new = [a_new(1),b_new(1),c_new(1),a_new(1)];
y_new = [a_new(2),b_new(2),c_new(2),a_new(2)];



% 内部点
x_int = 0.3;
y_int = 0.4;

figure(1)
subplot(121)
% 画出三角形和内部点
plot(x, y, 'k-', 'LineWidth', 2);
% 设置坐标轴范围
axis([-0.5 1.5 -0.5 1.5]);
hold on;
plot(x_int, y_int, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');




% 添加标题和标签
title('Triangle with an Interior Point');
xlabel('X');
ylabel('Y');
hold off
subplot(122)

% x y z 表示三角形的三个顶点
[alpha1,beta1,gamma1] = barycentric_coordinates(a(1),a(2),b(1),b(2),c(1),c(2),x_int,y_int)
[x_relative,y_relative] = relative_coordinates(a_new(1),a_new(2),b_new(1),b_new(2),c_new(1),c_new(2),alpha1,beta1,gamma1)

% 画出三角形和内部点
plot(x_new, y_new,'k-', 'LineWidth', 2);
% 设置坐标轴范围
axis([-0.5 1.5 -0.5 1.5]);
hold on;
plot(x_relative, y_relative, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');




% 添加标题和标签
title('Triangle with an Interior Point_New');
xlabel('X');
ylabel('Y');



function [alpha, beta, gamma] = barycentric_coordinates(x1, y1, x2, y2, x3, y3, x_int, y_int)
    % 计算三角形面积
    A = polyarea([x1, x2, x3], [y1, y2, y3]);

    % 计算内部点到三角形三个顶点的距离
    a = polyarea([x_int, x2, x3], [y_int, y2, y3]) / A;
    b = polyarea([x1, x_int, x3], [y1, y_int, y3]) / A;
    c = polyarea([x1, x2, x_int], [y1, y2, y_int]) / A;

    alpha = a;
    beta = b;
    gamma = c;
end

function [x_new, y_new] = relative_coordinates(x1, y1, x2, y2, x3, y3, alpha, beta, gamma)
    % 计算内部点的坐标
    x_new = alpha * x1 + beta * x2 + gamma * x3;
    y_new = alpha * y1 + beta * y2 + gamma * y3;

end
