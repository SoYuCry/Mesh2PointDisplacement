% 利用热像仪的图片得到Position的温度变化，需要调整bar
clc
clear all

position_X = 239;
position_Y = 149;
T_list =[];
t1 = [];
i = 52;
t = 0;
while(i-1<3367)
    I = imread(['jet-线性温标-固定\Rec-000031-084_07_46_28_619_',num2str(i),'.tif']);
    RGB = impixel(I,position_X,position_Y);
    if RGB(1)==0 & RGB(3)==255
        T = 15 + 4.5/255*RGB(2);
    end
    if RGB(1)==0 & RGB(2)==255
        T = 19.5 + 4.5/255*(255-RGB(3));
    end
    if RGB(3)==0 & RGB(2)==255
        T = 24 + 4.5/255*RGB(1);
    end
    if RGB(3)==0 & RGB(1)==255
        T = 28.5 + 4.5/255*(255-RGB(2));
    end
    T_list =[T_list,T];
    t1 =[t1,t];
    i = i + 6;
    t = t + 0.97136;
%     if i > 2566
%         imshow(I)
%     end
end

plot(t1,T_list)
hold on
t2 = [0:0.2:404.6];
T_list2 = load("Point_T_28.5.txt");
plot(t2,T_list2)


