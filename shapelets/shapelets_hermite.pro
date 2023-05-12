;$Id: shapelets_hermite.pro, v2$
;
; Copyright © 2005 Richard Massey and Alexandre Refregier.
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


function shapelets_hermite_coeffs, n

;
; NAME:
;       SHAPELETS_HERMITE_COEFFS
;
; CATEGORY:
;       A component of shapelets_hermite.pro.
;
; PURPOSE:
;       Compute the polynomial coefficients for Hn(x), the 1D Hermite
;       polynomial of order n
;
; INPUTS:
;       n - order of the Hermite polynomial
;
; KEYWORD PARAMETERS:
;       None.
;
; OUTPUTS:
;       h - vector of coefficients ci for Hn(x)=Sum_i ci*x^i, i=0,..,n
;
; MODIFICATION HISTORY:
;       Jan 09 - Common factors cancelled to enable 20<n<26 as integers by RM.
;       Jul 05 - Coefficients beyond n=20 calculated as double precisions by RM.
;       Sep 03 - Combined with shapelets_hermite.pro by Richard Massey.
;       Jul 99 - Written by A. Refregier
;

COMPILE_OPT idl2, HIDDEN

if n gt 25 then begin

  coeffs = dblarr(n+1) ; Resort to floating point arrays
  for i=0L,fix(n/2) do $
    coeffs[n-2*i] = factorial(n) / ( factorial(i) * factorial(n-2*i) ) * 2.d^(n-2*i) * (-1)^i

endif else begin

  coeffs = lon64arr(n+1)
  coeffs[n] = 2LL^n ; This works up to (and including) n=30

  if n gt 24 then begin

    integers=ul64indgen(n+1)
    for i=1L,fix(n/2) do begin
      numerator=integers[(i>(n-2*i))+1:n]
      denominator=integers[1:i<(n-2*i)>1]
      for j=n_elements(denominator)-1,1,-1 do begin ; no need to check j=0 since that will always be 1
        multiple=where(numerator mod denominator[j] eq 0,n_multiple)
        if n_multiple gt 0 then begin
          numerator[(reverse(multiple))[0]]=numerator[(reverse(multiple))[0]]/denominator[j]
          denominator[j]=1LL
        endif
      endfor
      coeffs[n-2*i] = product(numerator,/INTEGER)/product(denominator,/INTEGER) * 2ULL^(n-2*i) * (-1)^i ; This works up to (and including) n=25. Above that, the final coefficient does not fit into a 64 bit signed integer (max is 2ULL^63).
    endfor

  endif else if n gt 20 then begin

    integers=ul64indgen(n+1)
    for i=1L,fix(n/2) do $ ; This works up to (and including) n=24, but the first term overflows above that
      coeffs[n-2*i] = product(integers[(i>(n-2*i))+1:n],/INTEGER) / factorial(i<(n-2*i)>1,/UL64) * 2ULL^(n-2*i) * (-1)^i
    ; Note that the ratio of the first two terms really does always give an integer. I don't have a rigorous mathematical proof of this, but it does.

  endif else begin

    for i=1L,fix(n/2) do $ ; This works up to (and including) n=20, but 21! doesn't fit into a UL64 integer
      coeffs[n-2*i] = factorial(n,/UL64) / ( factorial(i,/UL64) * factorial(n-2*i,/UL64) ) * 2ULL^(n-2*i) * (-1)^i

  endelse
endelse

return, coeffs

end

; ----------------------------------------------------------------------------

function shapelets_hermite, n, x

;+
; NAME:
;       SHAPELETS_HERMITE
;
; CATEGORY:
;       Mathematical functions.
;
; PURPOSE:
;       Compute the 1D Hermite polynomial Hn(x) of order n.
;       Faster than the astlib HERMITE function because it stores a
;       pre-compiled set of low-order hermite functions.
;
;       Note that there are potentially rounding errors for large values of n 
;       and x, since the computation involves cancelling of large numbers.
;       However, this seems to be negligible in pratice.
;
; CALLING PROCEDURE:
;       result=shapelets_hermite(n,x)
;
; INPUTS:
;       n - order of the Hermite polynomial
;       x - can be a scalar, vector or array(coordinate grid)
;
; OUTPUTS:
;       Hn(x) - Hermite polynomial order n evaluated at position x
;               If x is of integer type, this will be a (longword/64bit) integer,
;               otherwise it is a (double precision) floating point.
;
; PROCEDURES USED:
;       shapelets_hermite_coeffs.
;
;       The version of IDL's "poly" routine built in below, takes advantage of 
;       the Hermite polynomials' alternating positive and negative coefficients 
;       to keep variables as low as possible, and therefore more significant 
;       figures in the final answer. This is true even though the coefficients
;       can be perfectly represented in signed longword integers up to n=27,
;       because the product of the coefficients and the x values is more likely
;       to be a (double precision) floating point, and if x is large, quite a
;       few significant figures could be compromised.
;
;       Never mind, poly is written in a very clever way which already makes
;       sure that any coefficient is never multiplied by x^n where n is large
;       without being summed along the way.
;
; MODIFICATION HISTORY:
;       Jan 09 - Sped up further by embedding poly & adding coeffs to 25 by RM.
;       Jul 05 - Sped up by RM using IDL's built-in poly function.
;       Oct 03 - Number of hardwired coeffs increased to 20 by R. Massey
;       Jul 99 - Written by A. Refregier
;-

COMPILE_OPT idl2

case n of
   0: return,x*0+1 ; Special case returns all 1s, of the same type and with the same array dimensions as x (NB: it used to return a floating point 1 if n=0)
   1: c = [0,2]
   2: c = [-2,0,4]
   3: c = [0,-12,0,8]
   4: c = [12,0,-48,0,16]
   5: c = [0,120,0,-160,0,32]
   6: c = [-120,0,720,0,-480,0,64]
   7: c = [0,-1680,0,3360,0,-1344,0,128]
   8: c = [1680,0,-13440,0,13440,0,-3584,0,256]
   9: c = [0,30240,0,-80640,0,48348,0,-9216,0,512]
  10: c = [-30240,0,302400,0,-403200,0,161280,0,-23040,0,1024]
  11: c = [0,-665280,0,2217600,0,-1774080,0,506880,0,-56320,0,2048]
  12: c = [665280,0,-7983360,0,13305600,0,-7096320,0,1520640,0,-135168,0,4096]
  13: c = [0,17297280,0,-69189120,0,69189120,0,-26357760,0,4392960,0,-319488,0,8192]
  14: c = [-17297280,0,242161920,0,-484323840,0,322882560,0,-92252160,0,12300288,0,-745472,0,16384]
  15: c = [0,-518918400,0,2421619200,0,-2905943040,0,1383782400,0,-307507200,0,33546240,0,-1720320,0,32768]
  16: c = [518918400,0,-8302694400,0,19372953600,0,-15498362880,0,5535129600,0,-984023040,0,89456640,0,-3932160,0,65536]
  17: c = [0,17643225600,0,-94097203200,0,131736084480,0,-75277762560,0,20910489600,0,-3041525760,0,233963520,0,-8912896,0,131072]
  18: c = [-17643225600,0,317578060800,0,-846874828800,0,790416506880,0,-338749931520,0,75277762560,0,-9124577280,0,601620480,0,-20054016,0,262144]
  19: c = [0,-670442572800,0,4022655436800,0,-6436248698880,0,4290832465920,0,-1430277488640,0,260050452480,0,-26671841280,0,1524105216,0,-44826624,0,524288]
  20: c = [670442572800,0,-13408851456000,0,40226554368000,0,-42908324659200,0,21454162329600,0,-5721109954560,0,866834841600,0,-76205260800,0,3810263040,0,-99614720,0,1048576]
  21: c = [0,28158588057600,0,-187723920384000,0,337903056691200,0,-257449947955200,0,100119424204800,0,-21844238008320,0,2800543334400,0,-213374730240,0,9413591040,0,-220200960,0,2097152]
  22: c = [-28158588057600,0,619488937267200,0,-2064963124224000,0,2477955749068800,0,-1415974713753600,0,440525466501120,0,-80095539363840,0,8801707622400,0,-586780508160,0,23011000320,0,-484442112,0,4194304]
  23: c = [0,-1295295050649600,0,9498830371430400,0,-18997660742860800,0,16283709208166400,0,-7237204092518400,0,1842197405368320,0,-283414985441280,0,26991903375360,0,-1587759022080,0,55710842880,0,-1061158912,0,8388608]
  24: c = [1295295050649600,0,-31087081215590400,0,113985964457164800,0,-151981285942886400,0,97702255248998400,0,-34738579644088320,0,7368789621473280,0,-971708521512960,0,80975710126080,0,-4234024058880,0,133706022912,0,-2315255808,0,16777216]
  25: c = [0,64764752532480000,0,-518118020259840000,0,1139859644571648000,0,-1085580613877760000,0,542790306938880000,0,-157902634745856000,0,28341498544128000,0,-3239028405043200,0,238163853312000,0,-11142168576000,0,318347673600,0,-5033164800,0,33554432]
else: c = shapelets_hermite_coeffs(n)
endcase

; A version of y=poly(x,c) built-in for speed, also using the fact that every other coefficient is zero
; Like poly, this is written in a clever way which already makes sure that any coefficient is never multiplied by x^n where n is large, without being summed along the way.
xsq = double(x)^2
y=c[n] & for i=n-2,0,-2 do y = temporary(y) * xsq + c[i]
if n mod 2 then y = temporary(y) * x

; Can we do it as exact integers?
if min(abs(size(x,/type)-[2,3,12,13,14,15])) eq 0 and max(y) lt 2.^62 then begin ; Max value in L64 integer is (2^63)-1 and leave a factor of 2 leeway
  xsq=long64(x)^2
  y=c[n] & for i=n-2,0,-2 do y = temporary(y) * xsq + c[i]
  if n mod 2 then y = temporary(y) * x
endif

;; Different version, which is 4x slower, but which I thought might be better at dealing with values of x>1 - start with the small terms, and knowing that the coefficients alternate in sign, thus nearly cancelling out in summation. In practice, this produces idential values.
;integers=size(c,/TYPE) ne 5
;odd=n mod 2
;xsqminus1=double(x^2)-1
;y=total(c,INTEGER=integers) 
;for i=2,n,2 do y = temporary(y) + total(c[i+odd:n],INTEGER=integers)*xsqminus1*double(x)^(i-2)
;if odd then y = temporary(y) * x ; Could actually be clever here and take out common factor of x differently for x<1 (include in each term) than x>1 (multiple by x at the end)

return, y

end
