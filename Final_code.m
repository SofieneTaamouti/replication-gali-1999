%% PREPARING THE WORKSPACE
clc;
clear;
clear session;
close all;
warning off all;

%% ADDING PATHS TO FIND FUNCTIONS
addpath(genpath("functions"));

%% LOADING AND CLEANING US DATA
%**************************************************************************
% Loading files
data1 = readtable("dataUS\hours.csv");
data2 = readtable("dataUS\employment.csv");
data3 = readtable("dataUS\m2.csv");
data4 = readtable("dataUS\cpi.csv");
data5 = readtable("dataUS\3mtb.csv");
data6 = readtable("dataUS\gdp.csv");
% Loading data
data7 = readtable("dataG7\gdp.csv");
data8 = readtable("dataG7\employment1.csv");
% extract columns
hours = data1.HOANBS(5:192);
emp_monthly = data2.CE16OV(1:564);
m_monthly = data3.MYAGM2USM052N(1:432);
p = data4.CPIAUCSL(145:576);
r = data5.TB3MS(301:732);
gdp = data6.GDPC1(5:192);
date= 1948.125:0.25:1994.875;

% Transform monthly data to quarterly
p = transpose(mean(reshape(p, 3, [])));
r = transpose(mean(reshape(r, 3, [])));
emp_monthly = reshape(emp_monthly,3,[]);
emp =  transpose(sum(emp_monthly));
m_monthly = reshape(m_monthly,3,[]);
m = transpose(sum(m_monthly));

% Taking logs
prod_hours_log = log(gdp) - log(hours);
prod_emp_log = log(gdp) - log(emp);
hours_log = log(hours);
emp_log = log(emp);
gdp_log = log(gdp);
m_log = log(m);
p_log = log(p);

%% **********LOADING AND CLEANING INTERNATIONAL DATA***********************
%**************************************************************************
% Extract columns
emp_log_ca = log(data8.Value(1:132));
emp_log_uk = log(data8.Value(397:491));
emp_log_ge = log(data8.Value(165:264));
emp_log_ja = log(data8.Value(265:396));
prod_log_ca = log(data7.Value(2:133)) - emp_log_ca;
prod_log_uk = log(data7.Value(698:792)) - emp_log_uk;
prod_log_ge = log(data7.Value(298:397)) - emp_log_ge;
prod_log_ja = log(data7.Value(530:661)) - emp_log_ja;

% Taking fd, detrend
emp_fd_ca = diff(emp_log_ca);
emp_fd_uk = diff(emp_log_uk);
emp_fd_ge = diff(emp_log_ge);
emp_fd_ja = diff(emp_log_ja);

prod_fd_ca = diff(prod_log_ca);
prod_fd_uk = diff(prod_log_uk);
prod_fd_ge = diff(prod_log_ge);
prod_fd_ja = diff(prod_log_ja);

%% UNIT ROOT AND COINTEGRATION TESTS
%**************************************************************************
% Unit root tests, level, US Data
dataUS_test1 = [hours_log emp_log prod_hours_log prod_emp_log gdp_log];
unit_root_test_lvl = zeros(5,2);
for i=1:5
    [h,pValue,stat,cValue] = adftest(dataUS_test1(:,i), Model="TS", Lags=4);
    unit_root_test_lvl(i,1) = stat;
    unit_root_test_lvl(i,2) = cValue;
end

% first-differences
hours_fd = diff(hours_log)*100;
prod_hours_fd = diff(prod_hours_log)*100;
prod_emp_fd = diff(prod_emp_log)*100;
emp_fd = diff(emp_log)*100;
gdp_fd = diff(gdp_log)*100;

% Unit root tests, first-differenced, US data
dataUS_test2 = [hours_fd emp_fd prod_hours_fd prod_emp_fd gdp_fd];
unit_root_test_fd = zeros(5,2);
for i=1:5
    [h,pValue,stat,cValue] = adftest(dataUS_test2(:,i), Model="TS", Lags=4);
    unit_root_test_fd(i,1) = stat;
    unit_root_test_fd(i,2) = cValue;
end

% Detrend labor
hours_model = fitlm(date, hours_log);
predic_hours = predict(hours_model);
hours_detrended = hours_log - predic_hours;

emp_model = fitlm(date, emp_log);
predic_emp = predict(emp_model);
emp_detrended = emp_log - predic_emp;

% Unit root tests, detrended labor, US data
dataUS_test3 = [hours_detrended emp_detrended];
unit_root_test_dt = zeros(2,2);
for i=1:2
    [h,pValue,stat,cValue] = adftest(dataUS_test2(:,i), Model="TS", Lags=4);
    unit_root_test_dt(i,1) = stat;
    unit_root_test_dt(i,2) = cValue;
end

%% FIRST BIVARIATE MODEL ESTIMATION: FIRST DIFFERENCED HOURS
%**************************************************************************
% ESTIMATE A BIVARIATE TO SELECT THE NUMBER OF LAGS

%**************************************************************************
%
X1 = [prod_hours_fd hours_fd];
datesnum1 = 1948.125:0.25:1994.875;

% Estimating the VAR
nlags1=2;
det1=1;
[VAR1, VARopt1] = VARmodel(X1,nlags1,det1);

% LONG RUN IDENTIFICATION AND IR COMPUTATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt1.ident = 'long';
VARopt1.nsteps = 12;
VARopt1.firstdate = datesnum1(1);
VARopt1.frequency = 'q';
% Compute IR
[IR1, VAR1] = VARir(VAR1,VARopt1);
% Compute error bands
[IRinf1,IRsup1,~,~] = VARirband(VAR1,VARopt1);

%% SECOND BIVARIATE MODEL ESTIMATION: FIRST DIFFERENCED EMPLOYMENT
%**************************************************************************
% % ESTIMATE A BIVARIATE TO SELECT THE NUMBER OF LAGS

%**************************************************************************
X2 = [prod_emp_fd emp_fd];
datesnum2 = 1948.125:0.25:1994.875;

% Estimating the VAR
nlags2=2;
det2=1;
[VAR2, VARopt2] = VARmodel(X2,nlags2,det2);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt2.ident = 'long';
VARopt2.nsteps = 12;
VARopt2.firstdate = datesnum2(1);
VARopt2.frequency = 'q';
% Compute IR
[IR2, VAR2] = VARir(VAR2,VARopt2);
% Compute error bands
[IRinf2,IRsup2,~,~] = VARirband(VAR2,VARopt2);

%% THIRD BIVARIATE MODEL ESTIMATION: DETRENDED HOURS
%**************************************************************************
datesnum3 = 1948.125:0.25:1994.875;
nlags3=2;
det3=1;

% Estimating the VAR
X3 = [prod_hours_fd hours_detrended(1:end-1)];
[VAR3, VARopt3] = VARmodel(X3,nlags3,det3);

% LONG RUN IDENTIFICATION
%**************************************************************************

% Options to get zero long-run restrictions set and compute IR
VARopt3.ident = 'long';
VARopt3.nsteps = 12;
VARopt3.firstdate = datesnum3(1);
VARopt3.frequency = 'q';
% Compute IR
[IR3, VAR3] = VARir(VAR3,VARopt3);
% Compute error bands
[IRinf3,IRsup3,~,~] = VARirband(VAR3,VARopt3);

%% FOURTH BIVARIATE MODEL ESTIMATION: DETRENDED EMPLOYMENT
%**************************************************************************
datesnum4 = 1948.125:0.25:1994.875;
nlags4=2;
det4=1;

% Estimating the VAR
X4 = [prod_emp_fd emp_detrended(1:end-1)];
[VAR4, VARopt4] = VARmodel(X4,nlags4,det4);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt4.ident = 'long';
VARopt4.nsteps = 12;
VARopt4.firstdate = datesnum4(1);
VARopt4.frequency = 'q';
% Compute IR
[IR4, VAR4] = VARir(VAR4,VARopt4);
% Compute error bands
[IRinf4,IRsup4,~,~] = VARirband(VAR4,VARopt4);

%% REPLICATING GALI (1999) TABLE 1
%**************************************************************************
% CORRELATION ESTIMATES - FIRST-DIFFERENCED HOURS (Panel A, figure 1)

%Unconditional correlation
corre1.uncond = corr(prod_hours_fd, hours_fd);

% Conditional correlation (Technology)
cov1 = sum(IR1(:,1,1).*IR1(:,2,1)); % conditional covariance 
var_prod1 = sum(IR1(:,1,1).^2); % conditional variance of productivity
var_hours1 = sum(IR1(:,2,1).^2); % conditional variance of hours
corre1.FDHours.condT = cov1/(sqrt(var_prod1*var_hours1)); % conditional correlation

% Conditional correlation (non-technology)
cov2 = sum(IR1(:,1,2).*IR1(:,2,2)); % conditional covariance 
var_prod2 = sum(IR1(:,1,2).^2); % conditional variance of productivity
var_hours2 = sum(IR1(:,2,2).^2); % conditional variance of hours
corre1.FDHours.condNT = cov2/(sqrt(var_prod2*var_hours2)); % conditional correlation

% CORRELATION ESTIMATES - FIRST-DIFFERENCED EMPLOYMENT

%Unconditional correlation
corre2.uncond = corr(prod_emp_fd, emp_fd);

% Conditional correlation (Technology)
cov3 = sum(IR2(:,1,1).*IR2(:,2,1)); % conditional covariance 
var_prod3 = sum(IR2(:,1,1).^2); % conditional variance of productivity
var_hours3 = sum(IR2(:,2,1).^2); % conditional variance of hours
corre2.FDHours.condT = cov3/(sqrt(var_prod3*var_hours3)); % conditional correlation

% Conditional correlation (non-technology)
cov4 = sum(IR2(:,1,2).*IR2(:,2,2)); % conditional covariance 
var_prod4 = sum(IR2(:,1,2).^2); % conditional variance of productivity
var_hours4 = sum(IR2(:,2,2).^2); % conditional variance of hours
corre2.FDHours.condNT = cov4/(sqrt(var_prod4*var_hours4)); % conditional correlation

% CORRELATION ESTIMATES - DETRENDED HOURS

%Unconditional correlation
corre3.uncond = corr(prod_hours_fd, hours_detrended(1:end-1));

% Conditional correlation (Technology)
cov5 = sum(IR3(:,1,1).*IR3(:,2,1)); % conditional covariance 
var_prod5 = sum(IR3(:,1,1).^2); % conditional variance of productivity
var_hours5 = sum(IR3(:,2,1).^2); % conditional variance of hours
corre3.FDHours.condT = cov5/(sqrt(var_prod5*var_hours5)); % conditional correlation

% Conditional correlation (non-technology)
cov6 = sum(IR3(:,1,2).*IR3(:,2,2)); % conditional covariance 
var_prod6 = sum(IR3(:,1,2).^2); % conditional variance of productivity
var_hours6= sum(IR3(:,2,2).^2); % conditional variance of hours
corre3.FDHours.condNT = cov6/(sqrt(var_prod6*var_hours6)); % conditional correlation

% CORRELATION ESTIMATES - DETRENDED EMPLOYMENT

%Unconditional correlation
corre4.uncond = corr(prod_emp_fd, emp_detrended(1:end-1));

% Conditional correlation (Technology)
cov7 = sum(IR4(:,1,1).*IR4(:,2,1)); % conditional covariance 
var_prod7 = sum(IR4(:,1,1).^2); % conditional variance of productivity
var_hours7 = sum(IR4(:,2,1).^2); % conditional variance of hours
corre4.FDHours.condT = cov7/(sqrt(var_prod7*var_hours7)); % conditional correlation

% Conditional correlation (non-technology)
cov8 = sum(IR4(:,1,2).*IR4(:,2,2)); % conditional covariance 
var_prod8 = sum(IR4(:,1,2).^2); % conditional variance of productivity
var_hours8 = sum(IR4(:,2,2).^2); % conditional variance of hours
corre4.FDHours.condNT = cov8/(sqrt(var_prod8*var_hours8)); % conditional correlation

%% REPLICATING FIGURE 1 GALI (1999)
%**************************************************************************
% ---------------------Plotting the data-----------------------------------
figure;
scatter(hours_fd, prod_hours_fd, 10, "black");
xlabel("hours")
ylabel("productivity")
title("Data")
saveas(gcf,'figures\figure1-first.jpeg');
saveas(gcf,'figures\figure1-first.eps');
close gcf
%--------------Plotting the technology component---------------------------
hours_techcomponent = VAR1.Ft(2,2)*hours_fd(2:end-1) + ...
    VAR1.Ft(4,2)*hours_fd(1:end-2);
prod_techcomponent = VAR1.Ft(2,1)*prod_hours_fd(2:end-1) ...
    + VAR1.Ft(4,1)*prod_hours_fd(1:end-2);
figure;
scatter(hours_techcomponent,prod_techcomponent);
xlabel("hours")
ylabel("productivity")
title("Technology component")
saveas(gcf,'figures\figure1-second.jpeg');
saveas(gcf,'figures\figure1-second.eps');
close gcf
%--------------Plotting the nontechnology component------------------------
hours_nontechcomponent = VAR1.Ft(3,2)*hours_fd(2:end-1) ...
 + VAR1.Ft(5,2)*hours_fd(1:end-2);
prod_nontechcomponent = VAR1.Ft(3,1)*prod_hours_fd(2:end-1) ...
    + VAR1.Ft(5,1)*prod_hours_fd(1:end-2);
scatter(hours_nontechcomponent,prod_nontechcomponent);
xlabel("hours")
ylabel("productivity")
title("Nontechnology component")
saveas(gcf,'figures\figure1-third.jpeg');
saveas(gcf,'figures\figure1-third.eps');
close gcf
%% REPLICATING GALI (1999) FIGURE 2
%**************************************************************************
% IMPULSE RESPONSE FUNCTIONS
%**************************************************************************
% plot Cumulative Impulse Response 
figure;
% plot prod response to supply shock
subplot(3,2,1);
plot(cumsum(IR1(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf1(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup1(:,1,1)),'-o', Color = "black");
title('technology shock');
xlabel('productivity');
% plot prod response to nontechnology shock
subplot(3,2,2);
plot(cumsum(IR1(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf1(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup1(:,1,2)),'-o', Color = "black");
title('Nontechnology shock');
xlabel('productivity');
% Computing gdp responses to shocks
IR_GDP1 = IR1(:,1,:) + IR1(:,2,:);
IRinf_GDP1 = IRinf1(:,1,:) + IRinf1(:,2,:);
IRsup_GDP1 = IRsup1(:,1,:) + IRsup1(:,2,:);
% plot output response to technology shock
subplot(3,2,3);
plot(cumsum(IR_GDP1(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf_GDP1(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup_GDP1(:,1,1)),'-o', Color = "black");
xlabel('GDP');
% plot output response to demand shock
subplot(3,2,4);
plot(cumsum(IR_GDP1(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf_GDP1(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup_GDP1(:,1,2)),'-o', Color = "black");
xlabel('GDP');
subplot(3,2,5);
% plot hours response to technology shock
plot(cumsum(IR1(:,2,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf1(:,2,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup1(:,2,1)),'-o', Color = "black");
xlabel('hours');
% plot hours response to nontechnology shock
subplot(3,2,6);
plot(cumsum(IR1(:,2,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf1(:,2,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup1(:,2,2)),'-o', Color = "black");
xlabel('hours');
saveas(gcf,'figures\figure2.jpeg'); % to save in a "figures" folder
saveas(gcf,'figures\figure2.eps'); % to save in a "figures" folder
close gcf

%% REPLICATING GALI (1999) FIGURE 3
%**************************************************************************
% IMPULSE RESPONSE FUNCTIONS
%**************************************************************************
% Plot Impulse Response
%VARirplot(IRbar,VARopt,IRinf,IRsup);
% Plot Cumulative Impulse Response 
figure;
% Plot supply shock
subplot(3,2,1);
plot(cumsum(IR3(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf3(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup3(:,1,1)),'-o', Color = "black");
title('Technology shock');
xlabel('productivity');
%
subplot(3,2,2);
plot(cumsum(IR3(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf3(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup3(:,1,2)),'-o', Color = "black");
title('Nontechnology shock');
xlabel('productivity');
% GDP, technology shock
IR_GDP3 = IR3(:,1,:) + IR3(:,2,:);
IRinf_GDP3 = IRinf3(:,1,:) + IRinf3(:,2,:);
IRsup_GDP3 = IRsup3(:,1,:) + IRsup3(:,2,:);
subplot(3,2,3);
plot(cumsum(IR_GDP3(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf_GDP3(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup_GDP3(:,1,1)),'-o', Color = "black");
xlabel('gdp');
% GDP, demand shock
subplot(3,2,4);
plot(cumsum(IR_GDP3(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf_GDP3(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup_GDP3(:,1,2)),'-o', Color = "black");
xlabel('gdp');
subplot(3,2,5);
% hours, tech shock
plot(cumsum(IR3(:,2,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf3(:,2,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup3(:,2,1)),'-o', Color = "black");
xlabel('hours');
% hours, nontech shock
subplot(3,2,6);
plot(cumsum(IR3(:,2,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf3(:,2,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup3(:,2,2)),'-o', Color = "black");
xlabel('hours');
saveas(gcf,'figures\figure3.jpeg');
saveas(gcf,'figures\figure3.eps');
close gcf
%% UNIT ROOT TESTS FOR THE FIVE VARIABLES MODEL

mp = m_log-p_log;

dataUS_test4 = [m_log p_log mp r];
unit_root_test_5var_lvl = zeros(4,2);
for i=1:4
    [h,pValue,stat,cValue] = adftest(dataUS_test4(:,i), Model="TS", Lags=4);
   unit_root_test_5var_lvl(i,1) = stat;
  unit_root_test_5var_lvl(i,2) = cValue;
end

m_fd = diff(m_log);
p_fd = diff(p_log);
mp_fd = diff(mp);
r_fd =  diff(r);
rp = r(1:end-1,1) - p_fd;
rp_fd = diff(rp);
p_2fd = diff(p_fd);
m_2fd = diff(m_fd);
dataUS_test5 = [m_fd p_fd mp_fd r_fd rp];
unit_root_test_5var_fd = zeros(5,2);
for i=1:5
    [h,pValue,stat,cValue] = adftest(dataUS_test5(:,i), Model="TS", Lags=4);
   unit_root_test_5var_fd(i,1) = stat;
  unit_root_test_5var_fd(i,2) = cValue;
end

[~, ~, stat_rp, cValue_rp] = adftest(rp);
[~, ~, stat_rp_fd, cValue_rp_fd] = adftest(rp_fd);

%% FIVE VARIABLES MODEL ESTIMATION: GROWTH RATES HOURS
%**************************************************************************
datesnum5 = 1959.125:0.25:1994.875;
nlags5=2;
det5=1;

% Estimating the VAR
X5 = [prod_hours_fd(45:end-1) hours_fd(45:end-1) mp_fd(1:end-1) rp_fd p_2fd];
[VAR5, VARopt5] = VARmodel(X5,nlags5,det5);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt5.ident = 'long';
VARopt5.nsteps = 12;
VARopt5.firstdate = datesnum5(1);
VARopt5.frequency = 'q';
% Compute IR
[IR5, VAR5] = VARir(VAR5,VARopt5);
% Compute error bands
[IRinf5,IRsup5,~,~] = VARirband(VAR5,VARopt5);

%% FIVE VARIABLES MODEL ESTIMATION: GROWTH RATES EMPLOYMENT
%**************************************************************************
datesnum6 = 1959.125:0.25:1994.875;
nlags6=2;
det6=1;

% Estimating the VAR
X6 = [prod_emp_fd(45:end-1) emp_fd(45:end-1) mp_fd(1:end-1) rp_fd p_2fd];
[VAR6, VARopt6] = VARmodel(X6,nlags6,det6);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt6.ident = 'long';
VARopt6.nsteps = 12;
VARopt6.firstdate = datesnum6(1);
VARopt6.frequency = 'q';
% Compute IR
[IR6, VAR6] = VARir(VAR6,VARopt6);
% Compute error bands
[IRinf6,IRsup6,~,~] = VARirband(VAR6,VARopt6);

%% FIVE VARIABLES MODEL ESTIMATION: DETRENDED HOURS
%**************************************************************************
datesnum7 = 1959.125:0.25:1994.875;
nlags7=2;
det7=1;

% Estimating the VAR
X7 = [prod_hours_fd(45:end-1) hours_detrended(45:end-2) mp_fd(1:end-1) ...
    rp_fd p_2fd];
[VAR7, VARopt7] = VARmodel(X7,nlags7,det7);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt7.ident = 'long';
VARopt7.nsteps = 12;
VARopt7.firstdate = datesnum7(1);
VARopt7.frequency = 'q';
% Compute IR
[IR7, VAR7] = VARir(VAR7,VARopt7);
% Compute error bands
[IRinf7,IRsup7,~,~] = VARirband(VAR7,VARopt7);

%% FIVE VARIABLES MODEL ESTIMATION: DETRENDED EMPLOYMENT
%*************************************************************************
datesnum8 = 1959.125:0.25:1994.875;
nlags8=2;
det8=1;

% Estimating the VAR
X8 = [prod_hours_fd(45:end-1) emp_detrended(45:end-2) mp_fd(1:end-1) ...
    rp_fd p_2fd];
[VAR8, VARopt8] = VARmodel(X8,nlags8,det8);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt8.ident = 'long';
VARopt8.nsteps = 12;
VARopt8.firstdate = datesnum8(1);
VARopt8.frequency = 'q';
% Compute IR
[IR8, VAR8] = VARir(VAR8,VARopt8);
% Compute error bands
[IRinf8,IRsup8,~,~] = VARirband(VAR8,VARopt8);

%% FIVE VARIABLES MODEL ESTIMATION: ALTERNATIVE SPECIFICATION
%---------------------first-differences hours------------------------------
%**************************************************************************
 datesnum9 = 1959.125:0.25:1994.875;
 nlags9=2;
 det9=1;

% Estimating the VAR
X9 = [prod_hours_fd(45:end-1) hours_fd(45:end-1) m_2fd ...
     r_fd(1:end-1) p_2fd];
[VAR9, VARopt9] = VARmodel(X9,nlags9,det9);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt9.ident = 'long';
VARopt9.nsteps = 12;
VARopt9.firstdate = datesnum9(1);
VARopt9.frequency = 'q';
% Compute IR
[IR9, VAR9] = VARir(VAR9,VARopt9);
% Compute error bands
[IRinf9,IRsup9,~,~] = VARirband(VAR9,VARopt9);

%% FIVE VARIABLES MODEL ESTIMATION: ALTERNATIVE SPECIFICATION
%---------------------first-differences employment------------------------------
%**************************************************************************
 datesnum16 = 1959.125:0.25:1994.875;
 nlags16=2;
 det16=1;

% Estimating the VAR
X16 = [prod_emp_fd(45:end-1) emp_fd(45:end-1) m_2fd ...
     r_fd(1:end-1) p_2fd];
[VAR16, VARopt16] = VARmodel(X16,nlags16,det16);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt16.ident = 'long';
VARopt16.nsteps = 12;
VARopt16.firstdate = datesnum16(1);
VARopt16.frequency = 'q';
% Compute IR
[IR16, VAR16] = VARir(VAR16,VARopt16);
% Compute error bands
[IRinf16,IRsup16,~,~] = VARirband(VAR16,VARopt16);

%% FIVE VARIABLES MODEL ESTIMATION: ALTERNATIVE SPECIFICATION
%--------------------detrended hours------------------------------
%**************************************************************************
 datesnum17 = 1959.125:0.25:1994.875;
 nlags17=2;
 det17=1;

% Estimating the VAR
X17 = [prod_hours_fd(45:end-1) hours_detrended(45:end-2) m_2fd ...
     r_fd(1:end-1) p_2fd];
[VAR17, VARopt17] = VARmodel(X17,nlags17,det17);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt17.ident = 'long';
VARopt17.nsteps = 12;
VARopt17.firstdate = datesnum17(1);
VARopt17.frequency = 'q';
% Compute IR
[IR17, VAR17] = VARir(VAR17,VARopt17);
% Compute error bands
[IRinf17,IRsup17,~,~] = VARirband(VAR17,VARopt17);

%% FIVE VARIABLES MODEL ESTIMATION: ALTERNATIVE SPECIFICATION
%---------------------DETRENDED EMP------------------------------
%**************************************************************************
 datesnum18 = 1959.125:0.25:1994.875;
 nlags18=2;
 det18=1;

% Estimating the VAR
X18 = [prod_emp_fd(45:end-1) emp_detrended(45:end-2) m_2fd ...
     r_fd(1:end-1) p_2fd];
[VAR18, VARopt18] = VARmodel(X18,nlags18,det18);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt18.ident = 'long';
VARopt18.nsteps = 12;
VARopt18.firstdate = datesnum9(1);
VARopt18.frequency = 'q';
% Compute IR
[IR18, VAR18] = VARir(VAR18,VARopt18);
% Compute error bands
[IRinf18,IRsup18,~,~] = VARirband(VAR18,VARopt18);

%% REPLICATING GALI (1999) TABLE 2
%**************************************************************************

% CORRELATION ESTIMATES - FIVE VARIABLE: FIRST-DIFFERENCED HOURS 

% Conditional correlation (Technology)
cov9 = sum(IR5(:,1,1).*IR5(:,2,1)); % conditional covariance 
var_prod9 = sum(IR5(:,1,1).^2); % conditional variance of productivity
var_hours9 = sum(IR5(:,2,1).^2); % conditional variance of hours
corre5.FDHours.condT = cov9/(sqrt(var_prod9*var_hours9)); % conditional correlation

% Conditional correlation (non-technology)
cov10 = sum(IR5(:,1,2).*IR5(:,2,2)) + sum(IR5(:,1,3)*sum(IR5(:,2,3))) ... 
    + sum(IR5(:,1,4)*sum(IR5(:,2,4))) + sum(IR5(:,1,5)*sum(IR5(:,2,5))); % conditional covariance 
var_prod10 = sum(IR5(:,1,2).^2) + sum(IR5(:,1,3).^2) + ...
    sum(IR5(:,1,4).^2) + sum(IR5(:,1,5).^2); % conditional variance of productivity
var_hours10 = sum(IR5(:,2,2).^2) + sum(IR5(:,2,3).^2) + ...
    sum(IR5(:,2,4).^2) + sum(IR5(:,2,5).^2); % conditional variance of hours
corre5.FDHours.condNT = cov10/(sqrt(var_prod10*var_hours10)); % conditional correlation

% CORRELATION ESTIMATES - FIRST-DIFFERENCED EMPLOYMENT

% Conditional correlation (Technology)
cov11 = sum(IR6(:,1,1).*IR6(:,2,1)); % conditional covariance 
var_prod11 = sum(IR6(:,1,1).^2); % conditional variance of productivity
var_hours11 = sum(IR6(:,2,1).^2); % conditional variance of hours
corre6.FDHours.condT = cov11/(sqrt(var_prod11*var_hours11)); % conditional correlation

% Conditional correlation (non-technology)
cov12 = sum(IR6(:,1,2).*IR6(:,2,2)) + sum(IR6(:,1,3)*sum(IR6(:,2,3))) ... 
    + sum(IR6(:,1,4)*sum(IR6(:,2,4))) + sum(IR6(:,1,5)*sum(IR6(:,2,5))); % conditional covariance 
var_prod12 = sum(IR6(:,1,2).^2) + sum(IR6(:,1,3).^2) + ...
    sum(IR6(:,1,4).^2) + sum(IR6(:,1,5).^2); % conditional variance of productivity
var_hours12 = sum(IR6(:,2,2).^2) + sum(IR6(:,2,3).^2) + ...
    sum(IR6(:,2,4).^2) + sum(IR6(:,2,5).^2); % conditional variance of hours
corre6.FDHours.condNT = cov12/(sqrt(var_prod12*var_hours12)); % conditional correlation

% CORRELATION ESTIMATES - DETRENDED HOURS

% Conditional correlation (Technology)
cov13 = sum(IR7(:,1,1).*IR7(:,2,1)); % conditional covariance 
var_prod13 = sum(IR7(:,1,1).^2); % conditional variance of productivity
var_hours13 = sum(IR7(:,2,1).^2); % conditional variance of hours
corre7.FDHours.condT = cov13/(sqrt(var_prod13*var_hours13)); % conditional correlation

% Conditional correlation (non-technology)
cov14 = sum(IR7(:,1,2).*IR7(:,2,2)) + sum(IR7(:,1,3)*sum(IR7(:,2,3))) ... 
    + sum(IR7(:,1,4)*sum(IR7(:,2,4))) + sum(IR7(:,1,5)*sum(IR7(:,2,5))); % conditional covariance 
var_prod14 = sum(IR7(:,1,2).^2) + sum(IR7(:,1,3).^2) + ...
    sum(IR7(:,1,4).^2) + sum(IR7(:,1,5).^2); % conditional variance of productivity
var_hours14 = sum(IR7(:,2,2).^2) + sum(IR7(:,2,3).^2) + ...
    sum(IR7(:,2,4).^2) + sum(IR7(:,2,5).^2); % conditional variance of hours
corre7.FDHours.condNT = cov14/(sqrt(var_prod14*var_hours14)); % conditional correlation

% CORRELATION ESTIMATES - DETRENDED EMPLOYMENT

% Conditional correlation (Technology)
cov15 = sum(IR8(:,1,1).*IR8(:,2,1)); % conditional covariance 
var_prod15 = sum(IR8(:,1,1).^2); % conditional variance of productivity
var_hours15 = sum(IR8(:,2,1).^2); % conditional variance of hours
corre8.FDHours.condT = cov15/(sqrt(var_prod15*var_hours15)); % conditional correlation

% Conditional correlation (non-technology)
cov16 = sum(IR8(:,1,2).*IR8(:,2,2)) + sum(IR8(:,1,3)*sum(IR8(:,2,3))) ... 
    + sum(IR8(:,1,4)*sum(IR8(:,2,4))) + sum(IR8(:,1,5)*sum(IR8(:,2,5))); % conditional covariance 
var_prod16 = sum(IR8(:,1,2).^2) + sum(IR8(:,1,3).^2) + ...
    sum(IR8(:,1,4).^2) + sum(IR8(:,1,5).^2); % conditional variance of productivity
var_hours16 = sum(IR8(:,2,2).^2) + sum(IR8(:,2,3).^2) + ...
    sum(IR8(:,2,4).^2) + sum(IR8(:,2,5).^2); % conditional variance of hours
corre8.FDHours.condNT = cov16/(sqrt(var_prod16*var_hours16)); % conditional correlation

% Table
%% REPLICATING GALI (1999) FIGURE 4
%**************************************************************************
% IMPULSE RESPONSE FUNCTIONS
%**************************************************************************
% Plot Impulse Response
% VARirplot(IRbar,VARopt,IRinf,IRsup);
% Plot Cumulative Impulse Response 
figure;
% productivity response to technology shock
subplot(3,2,1)
plot(cumsum(IR5(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf5(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup5(:,1,1)),'-o', Color = "black")
xlabel('productivity')
% real balances response to technology shock
subplot(3,2,2)
plot(cumsum(IR5(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf5(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup5(:,3,1)),'-o', Color = "black")
xlabel('real balances')
% gdp response to technology shock
IR_GDP5 = IR5(:,1,:) + IR5(:,2,:);
IRinf_GDP5 = IRinf5(:,1,:) + IRinf5(:,2,:);
IRsup_GDP5 = IRsup5(:,1,:) + IRsup5(:,2,:);
subplot(3,2,3)
plot(cumsum(IR_GDP5(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf_GDP5(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup_GDP5(:,1,1)),'-o', Color = "black")
xlabel('gdp')
% real rate response to technology shock
subplot(3,2,4)
plot(cumsum(IR5(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf5(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup5(:,4,1)),'-o', Color = "black")
xlabel('real rate')
subplot(3,2,5)
% hours response to technology shock
plot(cumsum(IR5(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf5(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup5(:,2,1)),'-o', Color = "black")
xlabel('hours')
% inflation response to technology shock
subplot(3,2,6)
plot(cumsum(IR5(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf5(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup5(:,5,1)),'-o', Color = "black")
xlabel('inflation')
saveas(gcf,'figures\figure4.jpeg');
saveas(gcf,'figures\figure4.eps');
close gcf

%% UNIT ROOT TESTS FOR INTERNATIONAL DATA
%**************************************************************************

dataINT_test1 = [prod_log_ca emp_log_ca prod_log_ja emp_log_ja];
unit_root_test_int_lvl = zeros(4,2);
for i=1:4
    [h,pValue,stat,cValue] = adftest(dataINT_test1(:,i), ...
        Model="TS", Lags=4);
   unit_root_test_int_lvl(i,1) = stat;
  unit_root_test_int_lvl(i,2) = cValue;
end
 
[~,~,stat_ge,cValue_ge] = adftest(prod_log_ge, Model="TS", Lags=4);
[~,~,stat_uk,cValue_uk] = adftest(prod_log_uk, Model="TS", Lags=4);

 dataINT_test2 = [prod_fd_ca emp_fd_ca prod_fd_ja emp_fd_ja];
unit_root_test_int_fd = zeros(4,2);
for i=1:4
    [h,pValue,stat,cValue] = adftest(dataINT_test2(:,i), ...
        Model="TS", Lags=4);
   unit_root_test_int_fd(i,1) = stat;
  unit_root_test_int_fd(i,2) = cValue;
end

[~,~,stat_ge_fd,cValue_ge_fd] = adftest(prod_fd_ge, Model="TS", Lags=4);
[~,~,stat_uk_fd,cValue_uk_fd] = adftest(prod_fd_uk, Model="TS", Lags=4);    
%% BIVARIATE MODEL ESTIMATION: CANADA (GR EMPLOYMENT, GR PROD)
%**************************************************************************

datesnum10 = 1962.125:0.25:1994.875;
nlags10=2;
det10=1;

% Estimating the VAR
X10 = [prod_fd_ca emp_fd_ca];
[VAR10, VARopt10] = VARmodel(X10,nlags10,det10);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt10.ident = 'long';
VARopt10.nsteps = 12;
VARopt10.firstdate = datesnum10(1);
VARopt10.frequency = 'q';
% Compute IR
[IR10, VAR10] = VARir(VAR10,VARopt10);
% Compute error bands
[IRinf10,IRsup10,~,~] = VARirband(VAR10,VARopt10);

%% BIVARIATE MODEL ESTIMATION: UNITED KINGDOM (GR EMPLOYMENT, GR PROD)
%**************************************************************************

datesnum11 = 1971.125:0.25:1994.750;
nlags11=2;
det11=1;

% Estimating the VAR
X11 = [prod_fd_uk emp_fd_uk];
[VAR11, VARopt11] = VARmodel(X11,nlags11,det11);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt11.ident = 'long';
VARopt11.nsteps = 12;
VARopt11.firstdate = datesnum11(1);
VARopt11.frequency = 'q';
% Compute IR
[IR11, VAR11] = VARir(VAR11,VARopt11);

%% BIVARIATE MODEL ESTIMATION: GERMANY (GR EMPLOYMENT, GR PROD)
%**************************************************************************
datesnum12 = 1970.125:0.25:1994.875;
nlags12=2;
det12=1;

% Estimating the VAR
X12 = [prod_fd_ge emp_fd_ge];
[VAR12, VARopt12] = VARmodel(X12,nlags12,det12);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt12.ident = 'long';
VARopt12.nsteps = 12;
VARopt12.firstdate = datesnum12(1);
VARopt12.frequency = 'q';
% Compute IR
[IR12, VAR12] = VARir(VAR12,VARopt12);

%% BIVARIATE MODEL ESTIMATION: JAPAN (GR EMPLOYMENT, GR PROD)
%**************************************************************************

datesnum15 = 1962.125:0.25:1994.875;
nlags15=2;
det15=1;

% Estimating the VAR
X15 = [prod_fd_ja emp_fd_ja];
[VAR15, VARopt15] = VARmodel(X15,nlags15,det15);

% LONG RUN IDENTIFICATION
%**************************************************************************
% Options to get zero long-run restrictions set and compute IR
VARopt15.ident = 'long';
VARopt15.nsteps = 12;
VARopt15.firstdate = datesnum15(1);
VARopt15.frequency = 'q';
% Compute IR
[IR15, VAR15] = VARir(VAR15,VARopt15);

%% REPLICATING GALI (1999) TABLE 3
%**************************************************************************
% -------------CORRELATION ESTIMATES: INTERNATIONAL EVIDENCE--------------- 

% CANADA

% Unconditional correlation
corre9.uncond = corr(prod_fd_ca, emp_fd_ca);

% Conditional correlation (Technology)
cov17 = sum(IR10(:,1,1).*IR10(:,2,1)); % conditional covariance 
var_prod17 = sum(IR10(:,1,1).^2); % conditional variance of productivity
var_hours17 = sum(IR10(:,2,1).^2); % conditional variance of hours
corre9.FDHours.condT = cov17/(sqrt(var_prod17*var_hours17)); % conditional correlation

% Conditional correlation (non-technology)
cov18 = sum(IR10(:,1,2).*IR10(:,2,2)); % conditional covariance 
var_prod18 = sum(IR10(:,1,2).^2); % conditional variance of productivity
var_hours18 = sum(IR10(:,2,2).^2); % conditional variance of hours
corre9.FDHours.condNT = cov18/(sqrt(var_prod18*var_hours18)); % conditional correlation

% UK

% Unconditional correlation
corre10.uncond = corr(prod_fd_uk, emp_fd_uk);

% Conditional correlation (Technology)
cov19 = sum(IR11(:,1,1).*IR11(:,2,1)); % conditional covariance 
var_prod19 = sum(IR11(:,1,1).^2); % conditional variance of productivity
var_hours19 = sum(IR11(:,2,1).^2); % conditional variance of hours
corre10.FDHours.condT = cov19/(sqrt(var_prod19*var_hours19)); % conditional correlation

% Conditional correlation (non-technology)
cov20 = sum(IR11(:,1,2).*IR11(:,2,2)); % conditional covariance 
var_prod20 = sum(IR11(:,1,2).^2); % conditional variance of productivity
var_hours20 = sum(IR11(:,2,2).^2); % conditional variance of hours
corre10.FDHours.condNT = cov20/(sqrt(var_prod20*var_hours20)); % conditional correlation

% Germany

% Unconditional correlation
corre11.uncond = corr(prod_fd_ge, emp_fd_ge);

% Conditional correlation (Technology)
cov21 = sum(IR12(:,1,1).*IR12(:,2,1)); % conditional covariance 
var_prod21 = sum(IR12(:,1,1).^2); % conditional variance of productivity
var_hours21 = sum(IR12(:,2,1).^2); % conditional variance of hours
corre11.FDHours.condT = cov21/(sqrt(var_prod21*var_hours21)); % conditional correlation

% Conditional correlation (non-technology)
cov22 = sum(IR12(:,1,2).*IR12(:,2,2)); % conditional covariance 
var_prod22 = sum(IR12(:,1,2).^2); % conditional variance of productivity
var_hours22= sum(IR12(:,2,2).^2); % conditional variance of hours
corre11.FDHours.condNT = cov22/(sqrt(var_prod22*var_hours22)); % conditional correlation

% Japan

%Unconditional correlation
corre14.uncond = corr(prod_fd_ja, emp_fd_ja);

% Conditional correlation (Technology)
cov27 = sum(IR15(:,1,1).*IR15(:,2,1)); % conditional covariance 
var_prod27 = sum(IR15(:,1,1).^2); % conditional variance of productivity
var_hours27 = sum(IR15(:,2,1).^2); % conditional variance of hours
corre14.FDHours.condT = cov27/(sqrt(var_prod27*var_hours27)); % conditional correlation

% Conditional correlation (non-technology)
cov28 = sum(IR15(:,1,2).*IR15(:,2,2)); % conditional covariance 
var_prod28 = sum(IR15(:,1,2).^2); % conditional variance of productivity
var_hours28 = sum(IR15(:,2,2).^2); % conditional variance of hours
corre14.FDHours.condNT = cov28/(sqrt(var_prod28*var_hours28)); % conditional correlation

% Table

%% REPLICATING GALI (1999) FIGURE 5
%**************************************************************************
% IMPULSE RESPONSE FUNCTIONS
%**************************************************************************
% plot Cumulative Impulse Response: figure 5, first part 
figure;
% canada tech shock
subplot(2,2,1)
plot(cumsum(IR10(:,1,1)),'--', Color = "black")
hold on
plot(cumsum(IR10(:,2,1)), Color = "black")
title('Canada: technology shock')
% canada non tech shock
subplot(2,2,2)
plot(cumsum(IR10(:,1,2)),'--', Color = "black")
hold on
plot(cumsum(IR10(:,2,2)), Color = "black")
title('Canada: nontechnology shock')
subplot(2,2,3)
% uk tech shock
plot(cumsum(IR11(:,1,1)),'--', Color = "black")
hold on
plot(cumsum(IR11(:,2,1)), Color = "black")
title('U.K.: technology shock')
subplot(2,2,4)
% uk, demand shock
plot(cumsum(IR11(:,1,2)),'--', Color = "black")
hold on
plot(cumsum(IR11(:,2,2)), Color = "black")
title('UK.: technology shock')
saveas(gcf,'figures\figure5.jpeg');
saveas(gcf,'figures\figure5.eps');
close gcf
% Figure 5, continued
figure;
% hours, tech shock
subplot(2,2,1)
plot(cumsum(IR12(:,1,1)),'--', Color = "black")
hold on
plot(cumsum(IR12(:,2,1)), Color = "black")
title('Germany: technology shock')
% hours, nontech shock
subplot(2,2,2)
plot(cumsum(IR12(:,1,2)),'--', Color = "black")
hold on
plot(cumsum(IR12(:,2,2)), Color = "black")
title('Germany: nontechnology shock')
% japan, tech shock
subplot(2,2,3)
plot(cumsum(IR15(:,1,1)),'--', Color = "black")
hold on
plot(cumsum(IR15(:,2,1)), Color = "black")
title('Japan: technology shock')
% japan, nontech shock
subplot(2,2,4)
plot(cumsum(IR15(:,1,2)),'--', Color = "black")
hold on
plot(cumsum(IR15(:,2,2)), Color = "black")
title('Japan: nontechnology shock')
saveas(gcf,'figures\figure5-continued.jpeg');
saveas(gcf,'figures\figure5-continued.eps');
close gcf
%% REPLICATING GALI (1999) FIGURE 6
% **********BUSINESS CYCLES DECOMPOSITION USING HPFILTER*******************
% Detrending hours using HP
[~, hours_detrended_hp] = hpfilter(hours_log, 1600);
% Computing tech components using first bivariate 
GDP_techcompo = 0.5832 + VAR1.Ft(2,1)*prod_hours_fd(2:end-1) ...
+ VAR1.Ft(4,1)*prod_hours_fd(1:end-2) + VAR1.Ft(2,2)*hours_detrended_hp(2:end-2) ...
+ VAR1.Ft(4,2)*hours_detrended_hp(1:end-3); 
hours_detrended_techcompo = 0.5832 + VAR1.Ft(2,2)*hours_detrended_hp(2:end-2) ...
+ VAR1.Ft(4,2)*hours_detrended_hp(1:end-3);

% Computing nontech components using first bivariate 
GDP_nontechcompo = 0.5832 + VAR1.Ft(3,1)*prod_hours_fd(2:end-1) ...
+ VAR1.Ft(5,1)*prod_hours_fd(1:end-2) + VAR1.Ft(3,2)*hours_detrended_hp(2:end-2) ...
+ VAR1.Ft(5,2)*hours_detrended_hp(1:end-3); 
hours_detrended_nontechcompo = 0.5832 + VAR1.Ft(3,2)*hours_detrended_hp(2:end-2) ...
+ VAR1.Ft(5,2)*hours_detrended_hp(1:end-3);
    
% Plotting technology component
figure;
subplot(2,1,1)
    plot(GDP_techcompo)
    hold on;
    plot(hours_detrended_techcompo, '--')
    title("Technology Component (HP-filtered)")
    % Plotting nontechnology component
    subplot(2,1,2)
    plot(GDP_nontechcompo);
    hold on;
    plot(hours_detrended_nontechcompo, '--')
    legend("GDP", "Hours",'Location', 'southoutside')
    title("Nontechnology Component (HP-filtered)")
saveas(gcf,'figures\figure6.jpeg');
saveas(gcf,'figures\figure6.eps');
close gcf
%% REPLICATION GALI (1999) APPENDIX - TABLE A-1
%****************UNIT ROOT TESTS US DATA**********************************%
% To export tables to latex
table2latex(array2table(unit_root_test_lvl),'unit-root_us_lvl.tex')
table2latex(array2table(unit_root_test_fd),'unit-root_us_fd.tex')
table2latex(array2table(unit_root_test_5var_lvl),'unit-root-5var_lvl.tex')
table2latex(array2table(unit_root_test_5var_fd),'unit-root-5var_fd.tex')


%% REPLICATION GALI (1999) APPENDIX - TABLE A-2
%****************UNIT ROOT TESTS INTERNATIONAL DATA***********************%
% To export tables to latex
table2latex(array2table(unit_root_test_int_lvl),'unit-root_int_lvl.tex')
table2latex(array2table(unit_root_test_int_fd),'unit-root_int_fd.tex')
disp(stat_uk)
disp(stat_uk_fd)
disp(stat_ge)
disp(stat_ge_fd)

%% REPLICATION GALI (1999) APPENDIX - FIGURE A-1
%**************PRODUCTIVITY EMPLOYMENT COMOVEMENTS************************%
% Plotting the data
figure;
scatter(emp_fd, prod_emp_fd);
saveas(gcf,'figures\figureA-1-first.appendix.jpeg');
saveas(gcf,'figures\figureA-1-first.appendix.eps');
close gcf

%--------------Plotting the technology component---------------------------
hours_techcomponent = VAR2.Ft(2,2)*hours_fd(2:end-1) + ...
    VAR2.Ft(4,2)*hours_fd(1:end-2);
prod_techcomponent = VAR2.Ft(2,1)*prod_hours_fd(2:end-1) ...
    + VAR2.Ft(4,1)*prod_hours_fd(1:end-2);
figure;
scatter(hours_techcomponent,prod_techcomponent);
xlabel("hours")
ylabel("productivity")
title("Technology component")
saveas(gcf,'figures\figureA-1-second.appendix.jpeg');
saveas(gcf,'figures\figureA-1-second.appendix.eps');
close gcf
%--------------Plotting the nontechnology component------------------------
hours_nontechcomponent = VAR2.Ft(3,2)*hours_fd(2:end-1) ...
 + VAR2.Ft(5,2)*hours_fd(1:end-2);
prod_nontechcomponent = VAR2.Ft(3,1)*prod_hours_fd(2:end-1) ...
    + VAR2.Ft(5,1)*prod_hours_fd(1:end-2);
scatter(hours_nontechcomponent,prod_nontechcomponent);
xlabel("hours")
ylabel("productivity")
title("Nontechnology component")
saveas(gcf,'figures\figureA-1-third.appendix.jpeg');
saveas(gcf,'figures\figureA-1-third.appendix.eps');
close gcf
%% REPLICATION GALI (1999) APPENDIX - FIGURE A-2
% IMPULSE RESPONSE FUNCTIONS
%**************************************************************************
% plot cumulative impulse Response 
figure;
% Plot supply shock
subplot(3,2,1);
plot(cumsum(IR2(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf2(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup2(:,1,1)),'-o', Color = "black");
title('technology shock');
xlabel('productivity');
% 
subplot(3,2,2);
plot(cumsum(IR2(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf2(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup2(:,1,2)),'-o', Color = "black");
title('transitory shock');
xlabel('productivity');
% GDP, technology shock
IR_GDP2 = IR2(:,1,:) + IR2(:,2,:);
IRinf_GDP2 = IRinf2(:,1,:) + IRinf2(:,2,:);
IRsup_GDP2 = IRsup2(:,1,:) + IRsup2(:,2,:);
subplot(3,2,3);
plot(cumsum(IR_GDP2(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf_GDP2(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup_GDP2(:,1,1)),'-o', Color = "black");
xlabel('gdp');
% GDP, demand shock
subplot(3,2,4);
plot(cumsum(IR_GDP2(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf_GDP2(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup_GDP2(:,1,2)),'-o', Color = "black");
xlabel('gdp');
subplot(3,2,5)
% hours, tech shock
plot(cumsum(IR2(:,2,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf2(:,2,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup2(:,2,1)),'-o', Color = "black");
xlabel('employment');
% hours, nontech shock
subplot(3,2,6);
plot(cumsum(IR2(:,2,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf2(:,2,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup2(:,2,2)),'-o', Color = "black");
xlabel('employment');
saveas(gcf,'figures\figureA-2.jpeg');
saveas(gcf,'figures\figureA-2.eps');
close gcf
%% REPLICATION GALI (1999) APPENDIX - FIGURE A-3
% IMPULSE RESPONSE FUNCTIONS
%**************************************************************************
% Plot Impulse Response
% VARirplot(IRbar,VARopt,IRinf,IRsup);
% Plot Cumulative Impulse Response 
figure;
% Plot supply shock
subplot(3,2,1);
plot(cumsum(IR4(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf4(:,1,1)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup4(:,1,1)),'-o', Color = "black");
title('technology shock');
xlabel('productivity');
%
subplot(3,2,2);
plot(cumsum(IR4(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRinf4(:,1,2)),'-o', Color = "black");
hold on;
plot(cumsum(IRsup4(:,1,2)),'-o', Color = "black");
title('transitory shock');
xlabel('productivity');
% GDP, technology shock
IR_GDP4 = IR4(:,1,:) + IR4(:,2,:);
IRinf_GDP4 = IRinf4(:,1,:) + IRinf4(:,2,:);
IRsup_GDP4 = IRsup4(:,1,:) + IRsup4(:,2,:);
subplot(3,2,3)
plot(cumsum(IR_GDP4(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf_GDP4(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup_GDP4(:,1,1)),'-o', Color = "black")
xlabel('gdp')
% GDP, demand shock
subplot(3,2,4)
plot(cumsum(IR_GDP4(:,1,2)),'-o', Color = "black")
hold on
plot(cumsum(IRinf_GDP4(:,1,2)),'-o', Color = "black")
hold on
plot(cumsum(IRsup_GDP4(:,1,2)),'-o', Color = "black")
xlabel('gdp')
subplot(3,2,5)
% hours, tech shock
plot(cumsum(IR4(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf4(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup4(:,2,1)),'-o', Color = "black")
xlabel('employment')
% hours, nontech shock
subplot(3,2,6)
plot(cumsum(IR4(:,2,2)),'-o', Color = "black")
hold on
plot(cumsum(IRinf4(:,2,2)),'-o', Color = "black")
hold on
plot(cumsum(IRsup4(:,2,2)),'-o', Color = "black")
xlabel('employment')
saveas(gcf,'figures\figureA-3.jpeg');
saveas(gcf,'figures\figureA-3.eps');
close gcf

%% REPLICATION GALI (1999) APPENDIX - FIGURES A-4b
% IMPULSE RESPONSE FUNCTIONS - US HOURS DETRENDED
%**************************************************************************
% Plot Impulse Response
% VARirplot(IRbar,VARopt,IRinf,IRsup);
% Plot Cumulative Impulse Response 
figure;
% productivity response to technology shock
subplot(3,2,1)
plot(cumsum(IR7(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf7(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup7(:,1,1)),'-o', Color = "black")
xlabel('productivity')
% real balances response to technology shock
subplot(3,2,2)
plot(cumsum(IR7(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf7(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup7(:,3,1)),'-o', Color = "black")
xlabel('real balances')
% gdp response to technology shock
IR_GDP7 = IR7(:,1,:) + IR7(:,2,:);
IRinf_GDP7 = IRinf7(:,1,:) + IRinf7(:,2,:);
IRsup_GDP7 = IRsup7(:,1,:) + IRsup7(:,2,:);
subplot(3,2,3)
plot(cumsum(IR_GDP7(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf_GDP7(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup_GDP7(:,1,1)),'-o', Color = "black")
xlabel('gdp')
% real rate response to technology shock
subplot(3,2,4)
plot(cumsum(IR7(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf7(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup7(:,4,1)),'-o', Color = "black")
xlabel('real rate')
subplot(3,2,5)
% hours response to technology shock
plot(cumsum(IR7(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf7(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup7(:,2,1)),'-o', Color = "black")
xlabel('hours')
% inflation response to technology shock
subplot(3,2,6)
plot(cumsum(IR7(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf7(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup7(:,5,1)),'-o', Color = "black")
xlabel('inflation')
saveas(gcf,'figures\figureA-4b.jpeg');
saveas(gcf,'figures\figureA-4b.eps');
close gcf
%% REPLICATION GALI (1999) APPENDIX - FIGURES A-4c
% Plot Impulse Response
% VARirplot(IRbar,VARopt,IRinf,IRsup);
% Plot Cumulative Impulse Response 
figure;
% productivity response to technology shock
subplot(3,2,1)
plot(cumsum(IR6(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf6(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup6(:,1,1)),'-o', Color = "black")
xlabel('productivity')
% real balances response to technology shock
subplot(3,2,2)
plot(cumsum(IR6(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf6(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup6(:,3,1)),'-o', Color = "black")
xlabel('real balances')
% gdp response to technology shock
IR_GDP6 = IR6(:,1,:) + IR6(:,2,:);
IRinf_GDP6 = IRinf6(:,1,:) + IRinf6(:,2,:);
IRsup_GDP6 = IRsup6(:,1,:) + IRsup6(:,2,:);
subplot(3,2,3)
plot(cumsum(IR_GDP6(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf_GDP6(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup_GDP6(:,1,1)),'-o', Color = "black")
xlabel('gdp')
% real rate response to technology shock
subplot(3,2,4)
plot(cumsum(IR6(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf6(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup6(:,4,1)),'-o', Color = "black")
xlabel('real rate')
subplot(3,2,5)
% hours response to technology shock
plot(cumsum(IR6(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf6(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup6(:,2,1)),'-o', Color = "black")
xlabel('hours')
% inflation response to technology shock
subplot(3,2,6)
plot(cumsum(IR6(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf6(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup6(:,5,1)),'-o', Color = "black")
xlabel('inflation')
saveas(gcf,'figures\figureA-4c.jpeg');
saveas(gcf,'figures\figureA-4c.eps');
close gcf

%% REPLICATION GALI (1999) APPENDIX - FIGURES A-4d
% Plot Impulse Response
% VARirplot(IRbar,VARopt,IRinf,IRsup);
% Plot Cumulative Impulse Response 
figure;
% productivity response to technology shock
subplot(3,2,1)
plot(cumsum(IR8(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf8(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup8(:,1,1)),'-o', Color = "black")
xlabel('productivity')
% real balances response to technology shock
subplot(3,2,2)
plot(cumsum(IR8(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf8(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup8(:,3,1)),'-o', Color = "black")
xlabel('real balances')
% gdp response to technology shock
IR_GDP8 = IR8(:,1,:) + IR8(:,2,:);
IRinf_GDP8 = IRinf8(:,1,:) + IRinf8(:,2,:);
IRsup_GDP8 = IRsup8(:,1,:) + IRsup8(:,2,:);
subplot(3,2,3)
plot(cumsum(IR_GDP8(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf_GDP8(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup_GDP8(:,1,1)),'-o', Color = "black")
xlabel('gdp')
% real rate response to technology shock
subplot(3,2,4)
plot(cumsum(IR8(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf8(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup8(:,4,1)),'-o', Color = "black")
xlabel('real rate')
subplot(3,2,5)
% hours response to technology shock
plot(cumsum(IR8(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf8(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup8(:,2,1)),'-o', Color = "black")
xlabel('hours')
% inflation response to technology shock
subplot(3,2,6)
plot(cumsum(IR8(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf8(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup8(:,5,1)),'-o', Color = "black")
xlabel('inflation')
saveas(gcf,'figures\figureA-4c.jpeg');
saveas(gcf,'figures\figureA-4c.eps');
close gcf

%% REPLICATION GALI (1999) APPENDIX - FIGURES A-4e
% IMPULSE RESPONSE FUNCTIONS
%**************************************************************************
% Plot Impulse Response
% VARirplot(IRbar,VARopt,IRinf,IRsup);
% Plot Cumulative Impulse Response 
figure;
% productivity response to technology shock
subplot(3,2,1)
plot(cumsum(IR9(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf9(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup9(:,1,1)),'-o', Color = "black")
xlabel('productivity')
% real balances response to technology shock
subplot(3,2,2)
plot(cumsum(IR9(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf9(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup9(:,3,1)),'-o', Color = "black")
xlabel('real balances')
% gdp response to technology shock
IR_GDP9 = IR9(:,1,:) + IR9(:,2,:);
IRinf_GDP9 = IRinf9(:,1,:) + IRinf9(:,2,:);
IRsup_GDP9 = IRsup9(:,1,:) + IRsup9(:,2,:);
subplot(3,2,3)
plot(cumsum(IR_GDP9(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf_GDP9(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup_GDP9(:,1,1)),'-o', Color = "black")
xlabel('gdp')
% real rate response to technology shock
subplot(3,2,4)
plot(cumsum(IR9(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf9(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup9(:,4,1)),'-o', Color = "black")
xlabel('real rate')
subplot(3,2,5)
% hours response to technology shock
plot(cumsum(IR9(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf9(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup9(:,2,1)),'-o', Color = "black")
xlabel('hours')
% inflation response to technology shock
subplot(3,2,6)
plot(cumsum(IR9(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf9(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup9(:,5,1)),'-o', Color = "black")
xlabel('inflation')
saveas(gcf,'figures\figureA-4e.jpeg');
saveas(gcf,'figures\figureA-4e.eps');
close gcf

%% REPLICATION GALI (1999) APPENDIX - FIGURES A-4f
% IMPULSE RESPONSE FUNCTIONS
%**************************************************************************
% Plot Impulse Response
% VARirplot(IRbar,VARopt,IRinf,IRsup);
% Plot Cumulative Impulse Response 
figure;
% productivity response to technology shock
subplot(3,2,1)
plot(cumsum(IR16(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf16(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup16(:,1,1)),'-o', Color = "black")
xlabel('productivity')
% real balances response to technology shock
subplot(3,2,2)
plot(cumsum(IR16(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf16(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup16(:,3,1)),'-o', Color = "black")
xlabel('real balances')
% gdp response to technology shock
IR_GDP16 = IR16(:,1,:) + IR16(:,2,:);
IRinf_GDP16 = IRinf16(:,1,:) + IRinf16(:,2,:);
IRsup_GDP16 = IRsup16(:,1,:) + IRsup16(:,2,:);
subplot(3,2,3)
plot(cumsum(IR_GDP16(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf_GDP16(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup_GDP16(:,1,1)),'-o', Color = "black")
xlabel('gdp')
% real rate response to technology shock
subplot(3,2,4)
plot(cumsum(IR16(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf16(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup16(:,4,1)),'-o', Color = "black")
xlabel('real rate')
subplot(3,2,5)
% hours response to technology shock
plot(cumsum(IR16(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf16(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup16(:,2,1)),'-o', Color = "black")
xlabel('hours')
% inflation response to technology shock
subplot(3,2,6)
plot(cumsum(IR16(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf16(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup16(:,5,1)),'-o', Color = "black")
xlabel('inflation')
saveas(gcf,'figures\figureA-4f.jpeg');
saveas(gcf,'figures\figureA-4f.eps');
close gcf

%% REPLICATION GALI (1999) APPENDIX - FIGURES A-4g
% IMPULSE RESPONSE FUNCTIONS
%**************************************************************************
% Plot Impulse Response
% VARirplot(IRbar,VARopt,IRinf,IRsup);
% Plot Cumulative Impulse Response 
figure;
% productivity response to technology shock
subplot(3,2,1)
plot(cumsum(IR17(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf17(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup17(:,1,1)),'-o', Color = "black")
xlabel('productivity')
% real balances response to technology shock
subplot(3,2,2)
plot(cumsum(IR17(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf17(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup17(:,3,1)),'-o', Color = "black")
xlabel('real balances')
% gdp response to technology shock
IR_GDP17 = IR17(:,1,:) + IR17(:,2,:);
IRinf_GDP17 = IRinf17(:,1,:) + IRinf17(:,2,:);
IRsup_GDP17 = IRsup17(:,1,:) + IRsup17(:,2,:);
subplot(3,2,3)
plot(cumsum(IR_GDP17(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf_GDP17(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup_GDP17(:,1,1)),'-o', Color = "black")
xlabel('gdp')
% real rate response to technology shock
subplot(3,2,4)
plot(cumsum(IR17(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf17(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup17(:,4,1)),'-o', Color = "black")
xlabel('real rate')
subplot(3,2,5)
% hours response to technology shock
plot(cumsum(IR17(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf17(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup17(:,2,1)),'-o', Color = "black")
xlabel('hours')
% inflation response to technology shock
subplot(3,2,6)
plot(cumsum(IR17(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf17(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup17(:,5,1)),'-o', Color = "black")
xlabel('inflation')
saveas(gcf,'figures\figureA-4g.jpeg');
saveas(gcf,'figures\figureA-4g.eps');
close gcf

%% REPLICATION GALI (1999) APPENDIX - FIGURES A-4h
% IMPULSE RESPONSE FUNCTIONS
%**************************************************************************
% Plot Impulse Response
% VARirplot(IRbar,VARopt,IRinf,IRsup);
% Plot Cumulative Impulse Response 
figure;
% productivity response to technology shock
subplot(3,2,1)
plot(cumsum(IR18(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf18(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup18(:,1,1)),'-o', Color = "black")
xlabel('productivity')
% real balances response to technology shock
subplot(3,2,2)
plot(cumsum(IR18(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf18(:,3,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup18(:,3,1)),'-o', Color = "black")
xlabel('real balances')
% gdp response to technology shock
IR_GDP18 = IR18(:,1,:) + IR18(:,2,:);
IRinf_GDP18 = IRinf18(:,1,:) + IRinf18(:,2,:);
IRsup_GDP18 = IRsup18(:,1,:) + IRsup18(:,2,:);
subplot(3,2,3)
plot(cumsum(IR_GDP18(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf_GDP18(:,1,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup_GDP18(:,1,1)),'-o', Color = "black")
xlabel('gdp')
% real rate response to technology shock
subplot(3,2,4)
plot(cumsum(IR18(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf18(:,4,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup18(:,4,1)),'-o', Color = "black")
xlabel('real rate')
subplot(3,2,5)
% hours response to technology shock
plot(cumsum(IR18(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf18(:,2,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup18(:,2,1)),'-o', Color = "black")
xlabel('hours')
% inflation response to technology shock
subplot(3,2,6)
plot(cumsum(IR18(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRinf18(:,5,1)),'-o', Color = "black")
hold on
plot(cumsum(IRsup18(:,5,1)),'-o', Color = "black")
xlabel('inflation')
saveas(gcf,'figures\figureA-4h.jpeg');
saveas(gcf,'figures\figureA-4h.eps');
close gcf
