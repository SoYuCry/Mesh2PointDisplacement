% 专门用于plot,配合main使用，运行完main后使用
plot(t1,T_list)
hold on
t2 = [0:0.2:404.6];
T_list2 = load("Point_T_28.5.txt");
plot(t2,T_list2)
T_list2 = load("Point_T_28.7.txt");
plot(t2,T_list2)
T_list2 = load("Point_T_29.txt");
plot(t2,T_list2)
legend('热像仪','1.5mm深度','1.3mm深度','1mm深度')