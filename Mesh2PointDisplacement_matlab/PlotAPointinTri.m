%%%%%在三角形内部画一个点%%%%%
% 给定一个点P，若在三角形内部，则可视化

% 定义三角形的三个顶点abc
a = [1,2];
b = [3,2];
c = [2,4];

% 画出三角形
x = [a(1) b(1) c(1) a(1)];%x(1)y(1)z(1)
y = [a(2) b(2) c(2) a(2)];%x(2)y(2)z(2)
plot(x,y,'-o')  % 使用plot函数


% 给定一个点P
p = [2 3];
hold on 
plot(p(1), p(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');%画内部点

%判断并打印结果
ans = inTri(a, b, c, p);
disp(['结果：',num2str(ans)]); % 最终输出为-1


function ans = inTri(a, b, c, p)
% 判断点p是否在三角形abc内部
ab = b-a;
ap = p-a;
bc = c-b;
bp = p-b;
ca = a-c;
cp = p-c;
if cross2(ab,ap)>0 && cross2(bc,bp)>0 && cross2(ca,cp)>0 % 在三角形内部
    ans = 1;
elseif cross2(ab,ap) * cross2(bc,bp) * cross2(ca,cp) == 0 % 在边上
    ans = 0;
else % 在三角形外部
    ans = -1;
end

end

% 定义二维向量叉乘
function cross_product = cross2(a, b)
    cross_product = a(1)*b(2)-a(2)*b(1);
end

