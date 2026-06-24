% fibonacci.m — Fibonacci sequence
n = 10;
f = zeros(1, n);
f(1) = 1;
f(2) = 1;
for i = 3:n
    f(i) = f(i-1) + f(i-2);
end
disp(f);
