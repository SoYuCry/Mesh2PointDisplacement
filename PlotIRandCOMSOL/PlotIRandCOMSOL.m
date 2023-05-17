% 打开文件（热像仪导出的文件）
fileID = fopen('422187.txt', 'r');

% 检查文件是否成功打开
if fileID == -1
    error('Cannot open the file.');
end

% 初始化一个空的cell数组来存储文本文件的每一行
lines = {};

% 逐行读取文本文件
currentLine = fgetl(fileID);
while ischar(currentLine)
    % 将当前行添加到cell数组中
    lines{end+1} = currentLine;
    % 读取下一行
    currentLine = fgetl(fileID);
end

% 关闭文件
fclose(fileID);

lines_size = size(lines)
i = 2
T_list9 =[];
while i < lines_size(2)+1
    % 使用正则表达式提取科学记数法数字
    pattern = '(\-?\d+(\.\d+)?[eE][\+\-]\d+)';
    matches = regexp(lines(i), pattern, 'match');
    
    % 将匹配结果转换为数字
    number = str2double(matches{1});
    T_list9 =[T_list9,number];
    i = i + 1;
end

% 用总的加热时长除以数据的个数，得到中间的步长[0：步长：总时长]
t1 = [0:0.1654:108];
plot(t1,T_list1)
hold on
plot(t1,T_list2)
plot(t1,T_list3)
plot(t1,T_list4)
plot(t1,T_list5)
plot(t1,T_list6)
plot(t1,T_list7)
plot(t1,T_list8)
plot(t1,T_list9)

% t2 = [0:0.166:108];
% T_list2 = load("Point_T_new.txt");
% plot(t2,T_list2)

% 下面代码用于读取多个深度的图像并画在同一个figure中
% plot(t2,T_list2)
% T_list2 = load("Point_T_28.7.txt");
% plot(t2,T_list2)
% T_list2 = load("Point_T_29.txt");
% plot(t2,T_list2)
% legend('热像仪','1.5mm深度','1.3mm深度','1mm深度')