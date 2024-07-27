clear
pth = 'D:\OneDrive - University of Rochester\files\PhD_file\Ziyang_Deep_learning\deep_learn_readout\code_mlp\RAW_DATA\tomo_train_240608\train';
cd 'D:\OneDrive - University of Rochester\files\PhD_file\Ziyang_Deep_learning\deep_learn_readout\code_mlp\RAW_DATA\tomo_train_240608\train';

%% IQ plot

scale_factor = 1;

data = importdata([pth '\test\ground\test_ground_2k.txt']); % Ground
data2 = importdata([pth '\test\excited\test_excited_2k.txt']); % Excited

data = importdata([pth '\test\halfpix\test_halfpix_2k.txt']); % Ground
data2 = importdata([pth '\test\halfpiy\test_halfpiy_2k.txt']); % Excited


fn = (20e6*1000e-9)/1000;
N = 1000;
%for n = 1:1:259;
% for i = 1:1:20;
I=[];Q=[];II=[];QQ=[];Ir=[];Qr=[];It=[];Qt=[];
n = 1:1:1000;
% Read data
for i = 1:1:2000
Ii = data.data(2*i-1,:);
Qi = data.data(2*i,:);
%cosf = 1*cos(2*pi*fn*n);
%sinf = 1*sin(2*pi*fn*n);
% Demodulation
yIc = (10*cos(2*pi*fn*n)).*Ii;
yIs= (10*sin(2*pi*fn*n)).*Ii;
yQs = (10*sin(2*pi*fn*n)).*Qi;
yQc = (10*cos(2*pi*fn*n)).*Qi;

% Take average
AveIi = (sum(yIc+yQs)/N);
AveQi = (sum(-yIs+yQc)/N);
%AveIi = sqrt(power(sum(yIc)/N,2)+power(sum(yIs)/N,2));
%AveQi = sqrt(power(sum(yQs)/N,2)+power(sum(yQc)/N,2));

I = [I;AveIi];
Q = [Q;AveQi];
Ia = mean(I);
Iq = mean(Q);
Id= var(I);
Qd= var(Q);
std=sqrt(Id+Qd);
end
for i = 1:1:2000
Ii = data2.data(2*i-1,:);
Qi = data2.data(2*i,:);
Y=fft(Ii);
P1 = abs(Y/1504);
Y2=fft(Qi);
P2 = abs(Y2/1504);
%cosf = 1*cos(2*pi*fn*n);
%sinf = 1*sin(2*pi*fn*n);
yIc = (10*cos(2*pi*fn*n)).*Ii;
yIs= (10*sin(2*pi*fn*n)).*Ii;
yQs = (10*sin(2*pi*fn*n)).*Qi;
yQc = (10*cos(2*pi*fn*n)).*Qi;
AveIi = (sum(yIc+yQs)/N);
AveQi = (sum(-yIs+yQc)/N);
%AveIi = sqrt(power(sum(yIc)/N,2)+power(sum(yIs)/N,2));
%AveQi = sqrt(power(sum(yQs)/N,2)+power(sum(yQc)/N,2));
II = [II;AveIi];
QQ = [QQ;AveQi];
IIa = mean(II);
IIq = mean(QQ);
IId= var(II);
QQd= var(QQ);
stdd=sqrt(IId+QQd);
end
r1=std*scale_factor;
r2=stdd*scale_factor;
r3=sqrt((r1^2+r2^2)/2);
para=[IIa*scale_factor-r2,IIq*scale_factor-r2,2*r2,2*r2];
para2=[Ia*scale_factor-r1,Iq*scale_factor-r1,2*r1,2*r1];
dis=sqrt((IIa*scale_factor-Ia*scale_factor)^2+(IIq*scale_factor-Iq*scale_factor)^2);
ratio=dis/(2*r1);
figure(17)
axis on;
axis equal;

h1 = scatter(I*scale_factor,Q*scale_factor,'red');
axis equal;
hold on
sz = 120;
scatter(Ia*scale_factor,Iq*scale_factor,sz,'m','d','LineWidth',3);
hold on
h2 = scatter(II*scale_factor,QQ*scale_factor,'blue');
hold on
sz = 120;
scatter(IIa*scale_factor,IIq*scale_factor,sz,'c','d','LineWidth',3);
hold on
rectangle('position',para,'curvature',[1 1],'EdgeColor','c');
hold on
rectangle('position',para2,'curvature',[1 1],'EdgeColor','m');
hold on



%h3 = scatter(Ir,Qr,'cyan');
%hold on
%h4 = scatter(It,Qt,'green');
%hold on
%h3 = scatter(III,QQQ,'green');
%hold on
legend([h1(1),h2(1)],'Ground state','Excited state');
%legend([h1(1),h2(1),h3(1),h4(1)],'on2','off2','on1','off1');
xlabel('I(arb. units)');
ylabel('Q(arb. units)');
set(gca,'linewidth',2,'fontsize',15,'fontname','Helvetica');

%% Processing the data

dd = d.test_ground_1k;

% Open a file to write
fileID = fopen('test\ground\test_ground_1k.txt','w');
% fileID = fopen('excited\train_excited_2k.txt','w');

% Loop over each row in the matrix
for i = 1:size(dd,1)
    if mod(i,2) == 1
        % 'I' row
        fprintf(fileID, 'I,');
    else
        % 'Q' row
        fprintf(fileID, 'Q,');
    end
    
    % Print the data values, comma-separated
    fprintf(fileID, '%f,', dd(i,1:end-1));
    fprintf(fileID, '%f\n', dd(i,end)); % Last element without trailing comma
    
    % Print an empty line after each data row
    fprintf(fileID, '\n');
end

% Close the file
fclose(fileID);
