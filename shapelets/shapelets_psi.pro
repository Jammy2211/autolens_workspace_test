function shapelets_psi, nm, x1, x2,              $
                        BETA=beta,               $
                        ELLIPTICITY=ellipticity, $
                        THETA_ZERO=theta_zero,   $
                        INTEGRATE=integrate,     $
                        PREFACTOR=prefactor,     $
                        ARRAY=array,             $
                        DIAMOND=diamond,         $
                        SILENT=silent

;$Id: shapelets_psi.pro, v1$
;
; Copyright � 2005 Richard Massey and Alexandre Refregier.
;
; This file is a part of the Shapelets analysis code.
; www.astro.caltech.edu/~rjm/shapelets/
;
; The Shapelets code is free software; you can redistribute it and/or
; modify it under the terms of the GNU General Public Licence as published
; by the Free Software Foundation; either version 2 of the Licence, or
; (at your option) any later version.
;
; The Shapelets code is distributed in the hope that it will be useful,
; but WITHOUT ANY WARRANTY; without even the implied warranty of
; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
; GNU General Public Licence for more details.
;
; You should have received a copy of the GNU General Public Licence
; along with the Shapelets code; if not, write to the Free Software
; Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA.
;
;+
; NAME:
;      SHAPELETS_PSI
;
; CATEGORY:
;      Shapelets.
;
; PURPOSE:
;      Compute (dimensionless) exponential polar shapelet basis functions
;      psi(r, phi) in 2D, based on Laguerre function polynomials. 
;      Dimensionful basis functions can be calculated with
;      shapelets_psi(n,m, x1/beta[,x2/beta])/beta.
;
; INPUTS:
;      NM         - Basis function order. Integer [n,m].
;      x1         - Grid of x coordinates (e.g. created by shapelets_make_xarr.pro).
;      x2         - Grid of y coordinates if a 2D basis function is required.
;      BETA       - Shapelet scale size beta.
;
; KEYWORD PARAMETERS:
;      /ARRAY     - Instead of just returning one basis function, if /ARRAY is
;                   set, the routine will return an array containing all of
;                   the basis functions up to those associated with n_max=n
;
; OUTPUTS:
;      The polar exponential shapelet basis function psi_n_m(r, phi)
;
; MODIFICATION HISTORY:
;      Jul 10 - Adapted to Joel's X-lets by R. Massey
;      Jun 10 - Written by B. Rowe


if keyword_set(integrate) then message,"Sorry, can't integrate exponential shapelets within pixels... maybe I could oversample, though...",/INFO,NOPRINT=silent
if n_elements(nm) ne 2 then message,"Exponential shapelets currently only exist in 2D form"
if nm[0] lt 0 then message,"n must be non-negative!"
if abs(nm[1]) gt nm[0] then message,"|m| must not be greater than n!"
if not keyword_set(beta) then beta=1
;if size(x1,/N_DIMENSIONS) lt 2 or min(size(x1,/DIMENSIONS)) le 1 then x1=replicate(x1, 1) ; Error catching for 1xn arrays

; Establish number of pixels
if not keyword_set(x1) or not keyword_set(x2) then begin ; Default x and y arrays.
  shapelets_make_xarr,[128,128],x1,x2
  x1=x1*beta/8.
  x2=x2*beta/8.
endif
n_pix=size(x1,/DIMENSIONS) & n_pix_x=n_pix[0]
if size(x1,/N_DIMENSIONS) gt 1 then n_pix_y=long(n_pix[1]) else n_pix_y=1

; Evaluate grid coordinate arrays
if keyword_set(ellipticity) then begin ; Make the grid elliptical if necessary
  if size(ellipticity,/TYPE) eq 6 then begin ; (a-b)/(a+b)
    e=[float(ellipticity),imaginary(ellipticity)]
  endif else if n_elements(ellipticity) ge 2 then begin
    e=[ellipticity[0],ellipticity[1]]
  endif else message,"Ellipticity format not recognised!"
  ;eta=alog(1-total(e^2))-alog(1+total(e^2)) ; From BJ02 eq (2.8), with atanh(z)=[ln(1+z)-ln(1-z)]/2
  eta=-2*atanh(total(e^2)) ; From BJ02 eq (2.8)
  theta=atan(e[1],e[0]) ; PA=theta/2
  x1prime = (cosh(eta/2)+cos(theta)*sinh(eta/2))*x1+(sin(theta)*sinh(eta/2))*x2 ; BJ02 eq (2.9)
  x2prime = (sin(theta)*sinh(eta/2))*x1+(cosh(eta/2)-cos(theta)*sinh(eta/2))*x2
  ;x1prime = (1-e[0])*x1   -e[1]*x2
  ;x2prime =   -e[1]*x1  +(1+e[0])*x2
  ;determinant=1.-total(e^2)
;  x1=x1prime;/determinant
;  x2=x2prime;/determinant
endif else begin
  x1prime=x1
  x2prime=x2
endelse
r     = sqrt(x1prime^2+x2prime^2)/beta
theta = atan(x2prime,x1prime)
if keyword_set(theta_zero) then theta=temporary(theta)-(theta_zero/!radeg) ; Rotate basis functions if necessary

; Establish empty arrays to contain the basis functions
if not keyword_set(array) then begin
  n_coeffs=1
  n=nm[0]
  m=nm[1]
endif else shapelets_make_nvec,nm[0],n,m,n_coeffs,/POLAR,/EXPONENTIAL
BasisF = complexarr(n_pix_x,n_pix_y,n_coeffs,/nozero)

; Calculate constant prefactors
if not keyword_set(prefactor) then begin
  prefactor=1./sqrt(2*!pi)/beta * (n+0.5)^(-1-abs(m)) * sqrt(factorial(n-abs(m))/float(2*n+1)/factorial(n+abs(m)))*(-1)^(n+m)
endif else begin
  n_coeffs_old=n_elements(prefactor)
  if n_coeffs gt n_coeffs_old then begin
    prefactor=[prefactor,fltarr(n_coeffs-n_coeffs_old)]
    prefactor[n_coeffs_old:*]=1./sqrt(2*!pi)/beta * (n[n_coeffs_old:*]+0.5)^(-1-abs(m[n_coeffs_old:*])) * $
                              sqrt(factorial(n[n_coeffs_old:*]-abs(m[n_coeffs_old:*]))/float(2*n[n_coeffs_old:*]+1)/factorial(n[n_coeffs_old:*]+abs(m[n_coeffs_old:*])))*(-1)^(n[n_coeffs_old:*]+m[n_coeffs_old:*])
  endif
endelse
; Evaluate basis functions in each pixel
if keyword_set(array) then begin
  n_max=nm[0]
  m_max=n_max/(1+keyword_set(diamond))
  ern=fltarr(n_pix_x,n_pix_y,n_max+1)       & for i=0,n_max do ern[*,*,i]=exp(-r / (2. * float(i) + 1.))
  eim=complexarr(n_pix_x,n_pix_y,2*m_max+1) & for i=-m_max,m_max do eim[*,*,i+m_max]=complex(cos(i*theta), -sin(i*theta))
  ram=fltarr(n_pix_x,n_pix_y,m_max+1)       & for i=0,m_max do ram[*,*,i]=r^i
  for i=0,n_coeffs-1 do if (n[i]+abs(m[i]))*keyword_set(diamond) gt n_max then BasisF[*,*,i]=complex(0.,0.) else $
      BasisF[*,*,i]=prefactor[i] * ern[*,*,n[i]] * ram[*,*,abs(m[i])] * shapelets_kummer(n[i], m[i], r) * eim[*,*,m_max+m[i]]
endif else begin
  for i=0,n_coeffs-1 do begin
    BasisF[*,*,i]=prefactor[i] * exp(-r / (2. * float(n[i]) + 1.)) * r^abs(m[i]) * shapelets_kummer(n[i], m[i], r) * complex(cos(m[i]*theta), -sin(m[i]*theta))
    ;BasisF[*,*,i]=prefactor[i] * exp(-r/(2*n[i]+1.)) * r^abs(m[i]) * laguerre(r/(n[i]+0.5),n[i]-abs(m[i]),2*abs(m[i]),/DOUBLE) * complex(cos(m[i]*theta),-sin(m[i]*theta))
  endfor
endelse

return, BasisF

end
