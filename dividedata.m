function [data_train,data_test] = dividedata(data,ntrain,ntest)

rng default
p = randperm(length(data));
for i=1:round(ntrain*length(data))
data_train{1,i} = data{1,p(i)};
end
for t=1:round(ntest*length(data))
data_test{1,t} = data{1,p(t)};
end

end
