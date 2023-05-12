pro shapelets_recomp, decomp, recomp, x1, x2, $
                      PSF=psf,        $
                      NRANGE=nrange,  $
                      MEMORY=memory,  $
                      TOP=top,        $
                      BOTTOM=bottom,  $
                      SILENT=silent,  $
                      COMPLEX=complex,$
                      NOOVER=noover,  $
                      SKY=sky

;$Id: shapelets_recomp.pro, v2$
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
;
;+
; NAME:
;      SHAPELETS_RECOMP
;
; CATEGORY:
;      Shapelets.
;
; PURPOSE:
;      Compute the recomposed image corresponding to a set of shapelet
;      coefficients calculated using shapelets_decomp.pro. The input decomp
;      structure also contains a few meta-parameters, including the shapelet
;      scale size beta, and whether or not the basis functions should be
;      integrated within pixels, or merely evaluated at the centre of each
;      pixel.
;
; INPUTS:
;      DECOMP    - structure produced by shapelets_decomp.pro
;
; OPTIONAL INPUTS:
;      PSF       - A PSF as a decomp structure type. The model is (re-)convolved
;                  with this PSF in shapelet space before it is pixellated. 
;      X1, X2    - Array of pixel positions, which override any specified in the
;                  decomp structure. Used for irregularly-spaced grids.
;      NRANGE    - Integer array [n_low,n_high]. Ignore coefficients with n 
;                  outsdide this range.
;      TOP       - Integer. Use only this number of (Cartesian) shapelet 
;                  coefficients during recomposition, selected as the ones with 
;                  the largest absolute values.
;      BOTTOM    - Integer. Use only this number of (Cartesian) shapelet 
;                  coefficients during recomposition, selected as the ones with 
;                  the smallest absolute values.
;
; KEYWORD PARAMETERS:
;      /NOOVER   - If set, do not oversample the basis functions.
;      /COMPLEX  - If set, return a complex array (complex part is numerical
;                  error for a Cartesian shapelet model, but can be populated
;                  with polar shapelets).
;      /SKY      - If set, include fit to sky background in reconstructed image.
;      /MEMORY   - Slightly slower but more memory-efficient algorithm for large
;                  image arrays or very high n_max objects.
;
; OUTPUTS:
;      RECOMP    - Reconstructed (2D floating point) image array.
;
; TO DO:
;      Could simplify sky replacement etc. by using shapelets_make_ls_matrix.pro
;
; MODIFICATION HISTORY:
;      Jul 10 - Exponential shapelets option and flexible xarrays added by RM.
;      Jan 09 - Backward compatability of elliptical shapelets improved by RM.
;      Jul 07 - Fixed to work with older versions of IDL by RM.
;      Apr 07 - Elliptical basis functions treated by RM.
;      Apr 05 - COMPLEX keyword added by RM.
;      Apr 04 - PSF reconvolution added by RM.
;      Nov 01 - Numerical integration of basis functions added by Richard Massey.
;      Feb 01 - Modified by AR to allow oversampling of the basis functions
;      Jul 99 - Written by Alexandre Refregier
;-

COMPILE_OPT idl2

; Reconvolve with PSF, if necessary
decomp_r=decomp
if keyword_set(psf) then shapelets_convolve,decomp_r,psf

; Eliminate top few coefficients, if requested
if keyword_set(top) then begin
  sorted=reverse(sort(abs(decomp_r.coeffs)))
  coeffs_top=abs(decomp_r.coeffs[sorted[top-1]])
  decomp_r.coeffs[where(abs(decomp_r.coeffs) lt coeffs_top)]=0.
endif

; Eliminate bottom few coefficients, if requested
if keyword_set(bottom) then begin
  sorted=sort(abs(decomp_r.coeffs))
  coeffs_bottom=abs(decomp_r.coeffs[sorted[bottom-1]])
  decomp_r.coeffs[where(abs(decomp_r.coeffs) gt coeffs_bottom)]=0.
endif

; Oversample, if necessary
if keyword_set(noover) or not tag_exist(decomp_r,"oversample") then ov=1 else ov=decomp_r.oversample
ovf=float(ov)

; Create x arrays
if not keyword_set(x1) or not keyword_set(x2) then $
shapelets_make_xarr, [decomp_r.n_pixels[0]*ov, decomp_r.n_pixels[1]*ov], x1, x2, x0=decomp_r.x*ovf

; Old version used to convert to Cartesian shapelet representation, but this is never really necessary as it's done inside shapelets_chi.pro
;if tag_exist(decomp_r,"polar") then if decomp_r.polar then $
;  shapelets_polar_convert,decomp_r,/CARTESIAN,SILENT=silent

; construct the basis functions
if tag_exist(decomp_r,"basis_ellipticity") then basis_ellipticity=decomp_r.basis_ellipticity else basis_ellipticity=0
if tag_exist(decomp_r,"theta_zero") then theta_zero=decomp_r.theta_zero else theta_zero=0
if tag_exist(decomp_r,"polar") then if decomp_r.polar then polar=1
if tag_exist(decomp_r,"profile") then if strupcase(decomp_r.profile) eq "EXPONENTIAL" then exponential=1
if keyword_set(exponential) then begin
  if keyword_set(polar) then begin
    Basis=shapelets_psi([decomp_r.n_max,0], x1/(decomp_r.beta*ovf), x2/(decomp_r.beta*ovf), /array,  $
          integrate=decomp_r.integrate, ellipticity=basis_ellipticity,theta_zero=theta_zero,SILENT=silent) $ 
          / (decomp_r.beta*ovf)
    n=decomp.n
  endif else begin
    message,"Cartesian exponential shapelets are not yet defined"
  endelse
endif else begin
  if keyword_set(polar) then begin
    Basis=shapelets_chi([decomp_r.n_max,0], x1/(decomp_r.beta*ovf), x2/(decomp_r.beta*ovf), /array,  $
          integrate=decomp_r.integrate, ellipticity=basis_ellipticity,theta_zero=theta_zero,SILENT=silent) $ 
          / (decomp_r.beta*ovf)
    n=decomp.n
  endif else begin
    Basis=shapelets_phi([decomp_r.n_max,0], x1/(decomp_r.beta*ovf), x2/(decomp_r.beta*ovf), /array,  $
          integrate=decomp_r.integrate, ellipticity=basis_ellipticity,theta_zero=theta_zero,SILENT=silent) $ 
          / (decomp_r.beta*ovf)
    n=decomp_r.n1+decomp_r.n2
  endelse
endelse

; Eliminate coeffs outside n_max range if requested
if keyword_set(nrange) then begin
  outside_n_range=where(n lt nrange[0] or n gt nrange[1],n_outside_n_range)
  if n_outside_n_range gt 0 then decomp_r.coeffs[outside_n_range]=0.
endif

; Reconstruct the image by summing the (weighted) basis functions
if keyword_set(memory) then begin ; Old, memory-efficient version
  recomp=decomp_r.coeffs[0]*Basis[*,*,0] & for i=1,decomp_r.n_coeffs-1 do recomp=temporary(recomp)+decomp_r.coeffs[i]*Basis[*,*,i]
endif else begin
  if decomp_r.n_coeffs eq 1 then begin
    recomp=decomp_r.coeffs[0]*Basis[*,*,0]
  endif else begin ; Faster (vector addition) version from Barney's shapelets_surface_brightness.pro
    coeffs = reform(decomp_r.coeffs,1,1,decomp_r.n_coeffs,/OVERWRITE)
    if size(coeffs,/TYPE) eq 6 or size(coeffs,/TYPE) eq 9 then begin
      coeffs=complex(rebin(float(coeffs),[size(x1,/DIMENSIONS),n_elements(coeffs)]),$
                 rebin(imaginary(coeffs),[size(x1,/DIMENSIONS),n_elements(coeffs)]))
    endif else coeffs = rebin(coeffs,[size(x1,/DIMENSIONS),n_elements(coeffs)])
    recomp=total(temporary(coeffs)*Basis,3)
  endelse
endelse

; Resample --- er, does this work?
if ov ne 1 then recomp=complex(rebin(float(recomp),decomp_r.n_pixels[0],decomp_r.n_pixels[1]),$
                               rebin(imaginary(recomp),decomp_r.n_pixels[0],decomp_r.n_pixels[1]))

; Add sky background
if keyword_set(sky) then begin                              ; 
  recomp=recomp+decomp_r.skyfit[0]                          ; Add sky background
  if n_elements(decomp_r.skyfit) gt 1 then begin            ; 
    recomp=recomp+decomp_r.skyfit[1]*x1/(decomp_r.beta*ovf) ; Add sky gradient
    recomp=recomp+decomp_r.skyfit[2]*x2/(decomp_r.beta*ovf) ;
  endif                                                     ; 
endif  
  
; Make array complex
if keyword_set(complex) then recomp=complex(recomp) else recomp=float(recomp)

end










