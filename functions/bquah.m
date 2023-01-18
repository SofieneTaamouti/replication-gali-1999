function [X1,X2,epsd,epsz,A0,B0,B] = bquah(x,p,nt)
%BQUAH       generates impulse responses for the 2 x 1 structural VMA
%
%                   X(t) = A(0)e(t)+A(1)e(t-1)+...    
%
%            with the restrictions that E e e'= I, and sum_j a11(j)=0
%            where a11(j) is the upper left hand entry of A(j).
%            [X1,X2,epsd,epsz] = bquah(X,p,nt) requires a data matrix 
%            with 2 columns, a choice for the number of lags p in the 
%            estimated VAR: (I-B1 L-...Bp L^p) X(t) = B0+ v(t),
%            and a choice nt determining the horizon for the 
%            impulse responses.  The responses to a ``demand shock''
%            are in X1 and the responses to a ``technology shock'' are
%            in X2.  The empirical e(t)'s are in epsd and epsz. 
%

%            Ellen McGrattan, 3-12-04


[T,n]        = size(x);
if (n~=2);
  error('Program bquah requires that x have 2 columns')
end;

%
% Run a VAR(p) with the data, estimate Omega, and demean series
%
[B,se,resid] = autoreg(x,p);
Omega        = resid'*resid/(T-p);
B0           = B(:,1);
BB           = zeros(n);
for i=1:p;
  BB         = BB+B(:,(i-1)*n+2:i*n+1);
end;
B            = B(:,2:n*p+1);

%
% Let X(t) = C(L) v(t) be the Wold MA
% Compute S = sum_j C(j)
%
S            = (eye(n)-BB)\eye(n);

%
% Impose identifying restrictions:
% A0*A0=Omega and upper left of S*A0 equals 0
%

f            =-S(1,2)/S(1,1);
lam          = (Omega(1,1)*Omega(2,2) -Omega(1,2)^2)/ ...
               (Omega(1,1)+f^2*Omega(2,2)-2*f*Omega(1,2));
a21          =-sqrt(lam);
a11          = f*a21;
a12          = sqrt(Omega(1,1)-f^2*lam);
tem          = Omega(1,2)-f*lam;
a22          = sqrt(Omega(2,2)-lam);
if tem<0;
  a22        =-a22;
end;
A0           = [a11,a12;a21,a22];
tem          = S*A0;
if max(max(abs(A0*A0'-Omega)))>1e-5 | abs(tem(1,1))>1e-5 | any(imag(A0));
  error('Problem with identifying assumptions')
end;

%
% Compute impulse responses:
% X1 are responses to ``demand shocks''
% X2 are responses to ``technology shock''
%
X1           = zeros(2,nt);
X2           = zeros(2,nt);
X1(:,1)      = A0*[1;0];
X2(:,1)      = A0*[0;1]; 
Clag         = [eye(2),zeros(2,2*(p-1))];
for i=2:nt
  C          = zeros(2);
  for j=1:p;
    C        = C+B(:,(j-1)*2+1:2*j)*Clag(:,(j-1)*2+1:2*j);
  end;
  X1(:,i)    = C*A0*[1;0];
  X2(:,i)    = C*A0*[0;1]; 
  Clag       = [C,Clag(:,1:2*(p-1))];
end;
X1           = X1';
X2           = X2';

tem          = inv(A0)*resid';
epsd         = tem(1,:)';
epsz         = tem(2,:)';
