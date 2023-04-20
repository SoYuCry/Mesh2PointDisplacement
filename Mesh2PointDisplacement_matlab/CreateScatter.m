% 构造一个1x1x1的散射体

clc 
clear all

X=zeros(8,3);
X([5:8,11,12,15,16,18,20,22,24])=1;
d=[1 2 4 3 1 5 6 8 7 5 6 2 4 8 7 3];
plot3(X(d,1),X(d,2),X(d,3));
hold on

n = 1000;
a = 1;
x = []
y = []
z = []
i = 0
j = 0
k = 0

% 这里决定了散射体的范围和密度
XRANGE = 1
XSTEP = 0.1
YRANGE = 1
YSTEP = 0.1
ZRANGE = 1
ZSTEP = 0.1

while k < XRANGE
    while j < YRANGE
        while i < ZRANGE
            x(end+1) = k;
            y(end+1) = j;
            z(end+1) = i;
            i = i + ZSTEP;
        end
        j = j + YSTEP;
        i = 0;
    end
    k = k + XSTEP;
    j = 0
end
scatter3(x,y,z,'.');

