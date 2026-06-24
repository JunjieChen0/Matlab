% plot_demo.m — Basic plotting
x = 0:0.1:6.28;
y = sin(x);
figure;
plot(x, y);
xlabel('x');
ylabel('sin(x)');
title('Sine Wave');
grid('on');
saveas(gcf, 'sine.png');
disp('Plot saved to sine.png');
