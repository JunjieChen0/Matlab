% func_handle_demo.m — Function handles and anonymous functions
f = @sin;
disp(f(0));
disp(f(1.5708));

g = @(x, y) x.^2 + y.^2;
disp(g(3, 4));

h = @cos;
disp(h(0));
