% control_flow.m — Control flow test
n = 10;
if n > 5
    disp('n is greater than 5');
else
    disp('n is 5 or less');
end

for i = 1:5
    disp(i);
end

j = 1;
while j <= 3
    disp(j);
    j = j + 1;
end
