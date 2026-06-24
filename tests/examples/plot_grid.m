% plot_grid.m — Plot with grid on (command syntax)
x = 0:0.1:2*pi;
y = cos(x);
figure;
plot(x, y);
xlabel('x');
ylabel('cos(x)');
title('Cosine Wave');
grid on;
saveas(gcf, 'cosine.png');
disp('Plot saved to cosine.png');
