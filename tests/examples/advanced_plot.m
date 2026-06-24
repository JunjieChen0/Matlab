% advanced_plot.m — Advanced plotting
x = linspace(0, 2*pi, 100);
y1 = sin(x);
y2 = cos(x);

figure;
plot(x, y1, 'r-', x, y2, 'b--');
xlabel('x');
ylabel('y');
title('Sine and Cosine');
legend('sin', 'cos');
grid on;
saveas(gcf, 'sincos.png');
disp('Plot saved to sincos.png');
