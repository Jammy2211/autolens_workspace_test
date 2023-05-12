function shapelets_decomp, IMAGE, BETA, N_MAX,                  $
                           CENTRE=centre,                       $
                           NAME=name,                           $
                           BASIS_ELLIPTICITY=basis_ellipticity, $
                           THETA_ZERO=theta_zero,               $
                           PSF=psf,                             $
                           NOISE_INPUT=noise_input,             $
                           MASK_INPUT=mask_input,               $
                           RECOMP=recomp,                       $
                           RESIDUAL=residual,                   $
                           OVERSAMPLE=oversample,               $
                           OVERLAP=overlap,                     $
                           INTEGRATE=integrate,                 $
                           MEMORY=memory,                       $
                           SKY=sky,                             $
                           NON1=non1,                           $
                           INVERSION=inversion,                 $
                           POLAR=polar,                         $
                           EXPONENTIAL=exponential,             $
                           DIAMOND=diamond,                     $
                           MUSHROOM=mushroom,                   $
                           REBIN_RESIDUAL=rebin_residual,       $
                           SMOOTH_RESIDUAL=smooth_residual,     $
                           WAVELET_RESIDUAL=wavelet_residual,   $
                           FULL_ERROR=full_error,               $
                           SILENT=silent,                       $
                           LS=ls,                               $
                           X0=x0

;$Id: shapelets_decomp.pro, v2$
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
;      SHAPELETS_DECOMP
;
; CATEGORY:
;      Shapelets.
;
; PURPOSE:
;      Decompose an image into a weighted sum of shapelet basis functions.
;
; INPUTS:
;      IMAGE     - Shapelets image/pstamp structure, or 2D image array.
;      BETA      - Shapelet basis function scale size.
;      N_MAX     - Maximum order of the basis coefficients used in model.
;
; OPTIONAL INPUTS:
;      PSF       - A decomp structure representing the local PSF, to be
;                  decomvolved.
;      CENTRE    - Coordinates of the centre of the basis functions.
;                  DEFAULT: from structure, or centre of the image.
;      NAME      - Name of object (string).
;                  DEFAULT: from structure, or empty.
;      NOISE     - Inverse variance map of pixels, or constant noise value.
;                  Assumes zero covariance between adjacent pixels.
;                  DEFAULT: from structure, or constant unity.
;      MASK      - Array the same size as images, with zero for good pixels
;                  and one for pixels to ignore (could also be achieved with
;                  the weight map.
;      BASIS_ELLI- Ellipticity of basis functions, in weak lensing notation:
;                  either [e1,e2] or complex(e1,e2), where e1=e*cos(2theta) and
;                  e2=e*sin(2theta), with e=(a-b)/(a+b) for major and minor axes
;                  and theta is the angle of the major axis, a/c/w from x axis.
;                  This only works for basis functions not integrated in pixels.
;                  DEFAULT: circular basis functions.
;      THETA_ZERO- Basis functions are rotated by this anlge, a/c/w from x axis.
;                  DEFAULT: zero for circular basis functions, and the major
;                           axis of the ellipse for elliptical ones.
;      OVERSAMPLE- Force an oversampling factor to evaluate the basis fns.
;                  the basis functions (set over=1 to get round(1/theta_min))
;                  DEFAULT: no oversampling
;      SKY       - 1: fit sky background with a constant value
;                  2: fit sky gradient with a plane
;                  DEFAULT: no sky subtraction.
;      REBIN_RESI- Bin up the residual image by this integer factor before 
;                  calculating chi squared.
;                  DEFAULT: residual image used in same pixel scale as input.
;      SMOOTH_RES- Smooth the residual image (using a boxcar average of this
;                  number of pixels) before calculating chi squared.
;                  DEFAULT: residual image returned in same form as the input.
;	      
; KEYWORD PARAMETERS:
;      POLAR     - Perform decomposition using polar shapelet basis fns.
;                  DEFAULT: Cartesian shapelet basis functions.
;      EXPONENTIA- Perform decomposition using exponential shapelet basis fns.
;                  DEFAULT: Gaussian shapelet basis functions.
;      OVERLAP   - Obtain coefficients via Fourier-style overlap integrals.
;                  DEFAULT: least-squares fitting.
;      FULL_ERROR- Return full covariance matrix of coefficients.
;                  DEFAULT: return error on coefficients in error variable.
;      INTEGRATE - Use basis functions integrated within pixels.
;                  DEFAULT: integration (for central pixel value, set int=0)
;      NON1      - n=1 coefficients are forced to be zero, Kuijken-like.
;                  DEFAULT: all the coefficients are fit.
;      INVERSION - Use a simple matrix inversion rather than SVD.
;                  DEFAULT: singular valued decomposition, as of March 07.
;      DIAMOND   - Alternate truncation scheme for shapelet basis functions.
;                  DEFAULT: usual tringular scheme, with n<=n_max
;      
;
; OUTPUTS:
;      DECOMP    - A shapelet decomp structure is returned, containing the
;                  evaluated shapelet coefficeints, meta-parameters and other
;                  information.
;
; OPTIONAL OUTPUTS
;      RECOMP    - Repixellated image of the shapelet model (still containing
;                  the sky background and convolved with the original PSF. To
;                  remove this, use shapelets_recomp,decomp,recomp).
;      RESIDUAL  - Residual image (after any treatments used to adjust chi^2).
;      WAVELET_RE- Wavelet decomposition of the residual (before adjustments).
;
; CALLING SEQUENCE:
;      decomp=shapelets_decomp(image, beta, n_max)
;
; KNOWN BUGS:
;      Overlap integrals give slightly different results for polar and
;      Cartesian shapelet basis functions. Results are identical when fitting,
;      and I'd've thought that they would also be for overlap. Should they?
;
; TO DO:
;      In the SVD, should singular values be set to zero or infinity?
;      Do something (anything!) with the full covariance matrix of errros.
;
; MODIFICATION HISTORY:
;      Aug 10 - RM disabled simultaneous truncation schemes to simplify code.
;      Jul 10 - RM added EXPONENTIAL option.
;      Jun 10 - RM added ability to supply PSF as a bitmap image.
;      Feb 10 - RM improved handling of NaNs and memory for large images.
;      Sep 07 - RM made sure noise variable is created despite any input format.
;      Apr 07 - RM added THETA_ZERO option.
;      Mar 07 - RM implemented fitting with elliptical basis functions.
;      Mar 07 - RM replaced matrix inversion with singular value decomposition.
;      Mar 07 - RM added wavelet analysis of residual (needs JLS's software).
;      Mar 07 - Joel Berge recorrected propagation of errors on polar coeffs.
;      Feb 07 - RM added RESIDUAL keyword, plus options to rebin or smooth the
;               residual before calculating chi squared.
;      Feb 07 - RM used no. of independent, non-oversampled pixels in chi2. 
;      Sep 05 - RM changed the routine from a procedure to a function.
;      Sep 05 - RM added POLAR and DIAMOND options.
;      Aug 05 - RM accepted images in a wider variety of input formats.
;      Aug 05 - RM internalised LS fitting matrix and reorganised whole routine
;               in an attempt to speed things up.
;      Jul 05 - RM incorporated external create_decomp routine.
;      Jan 05 - RM optimised gamma and n_max_gamma during PSF deconvolution.
;      Apr 04 - RM added ability to mask pixels (by setting weight map to zero)
;               but still calculate the correct number of DOFs in the fit
;      Sep 03 - RM changed name of routine to shapelet_decomp.pro
;      Jul 03 - RM added "name" element to decomp structure
;      Apr 02 - RM added PSF deconvolution
;      Mar 02 - AR computed Chi^2 and fixed normalisation. Also added
;               option to return the recomposed (still convolved) image
;      Dec 01 - RM implemented least squares fitting of basis fns to data
;      Nov 01 - RM added background subtraction and sky fitting
;      Nov 01 - Richard Massey added numerical integration of basis fns
;      Feb 01 - AR incorporated oversampling of the basis functions
;      Jul 00 - AR changed to order by n=n1+n2
;      Jul 99 - mk_decomp.pro written by Alexandre Refregier
;-

COMPILE_OPT idl2
;ON_ERROR,2


;
; Maintain backwards compatability  by checking for obsolescent keywords
;
if n_params() ne 3 or size(n_max,/TYPE) ge 4 then message,"Usage: decomp=shapelets_decomp(image,beta,n_max)"
if keyword_set(ls) then message,"Least-squares fitting now the default. Use /OVERLAP for linear overlap method."
if keyword_set(x0) then begin & message,"Keyword X0 is obsolescent. Please use CENTRE.",/INFO & centre=x0 & endif
if keyword_set(non1)+keyword_set(mushroom)+keyword_set(diamond)+(n_elements(n_max) gt 1) gt 1 then message,"Multiple, simultaneous truncation schemes are not currently supported."
if n_elements(n_max) gt 1 and keyword_set(polar) then message,"n1<>n2 truncation scheme requires Cartesian shapelets."
if keyword_set(full_error) then message,"FULL_ERROR option not yet coded up."


;
; Create an empty structure to contain the final answer
;
decomp=shapelets_create_decomp(max(n_max),beta=beta,polar=polar,exponential=exponential,basis_ellipticity=basis_ellipticity,theta_zero=theta_zero,integrate=integrate,name=name)
if (keyword_set(integrate) or keyword_set(exponential)) and abs(decomp.basis_ellipticity) ne 0. then begin
  message,"Cannot integrate elliptical/exponential shapelet basis functions within pixels!",/INFO,NOPRINT=silent
  decomp.integrate=0B
endif


;
; Parse input image
;
if shapelets_structure_type(image,message=message,/SILENT) then begin
  if strupcase(image.type) eq "IMAGE" or strupcase(image.type) eq "PSTAMP" then begin
    data=image.image
    if tag_exist(image,"pixel_scale") then image_pixel_scale=image.pixel_scale
    if not keyword_set(name) then name=image.name
    if keyword_set(noise_input) then begin
      noise=noise_input
    endif else if tag_exist(image,"noise") then begin
      noise=image.noise
    endif else begin
      noise=1;replicate(1,(size(data,/dimensions))[0],(size(data,/dimensions))[1])
    endelse
    if keyword_set(mask_input) then begin
      masked_pixels=where(mask_input eq 1 or finite(data) eq 0,n_masked_pixels)
    endif else if tag_exist(image,"mask") then begin
      masked_pixels=where(image.mask eq 1 or finite(data) eq 0,n_masked_pixels)
    endif else begin
      n_masked_pixels=0
    endelse
    if not keyword_set(centre) then if tag_exist(image,"xo") and tag_exist(image,"yo") then begin
      centre=[image.xo,image.yo]
    endif
  endif else message,"Cannot apply shapelet transform to a "+image.type+" structure!"
endif else if size(image,/n_dimensions) eq 2 then begin
  data=image
  if keyword_set(noise_input) then noise=noise_input else noise=1;replicate(1,(size(image,/dimensions))[0],(size(image,/dimensions))[1])
  if keyword_set(mask_input) then masked_pixels=where(mask_input eq 1 or finite(data) eq 0,n_masked_pixels) else n_masked_pixels=0
endif else if size(image,/n_dimensions) eq 1 then begin
  message,"One dimensional shapelet transform not yet implemented"
endif else message,"Input image format not recognised!"
; Mask out pixels by setting their weight to zero
if n_masked_pixels gt 0 then begin
  if n_elements(noise) le 1 then noise=replicate(noise,(size(data,/dimensions))[0],(size(data,/dimensions))[1]) ;message,"Using a mask also requires a full noise map!"
  noise[masked_pixels]=0
endif


;
; Parse input PSF
;
if keyword_set(psf) then begin
  if shapelets_structure_type(psf,message=message,/SILENT) then begin
    if strupcase(psf.type) eq "DECOMP" then begin
      shapelets_polar_convert,psf,/P2C,/SILENT  
    endif else if strupcase(psf.type) eq "IMAGE" or strupcase(image.type) eq "PSTAMP" then begin
      if keyword_set(overlap) then message,"Cannot perform deconvolution from a PSF using overlap integrals when PSF is supplied as an image!"
      if tag_exist(psf,"pixel_scale") then psf_pixel_scale=psf.pixel_scale
      psf_image=psf.image
    endif else message,"Cannot convolve with a "+image.type+" structure!"
  endif else if size(psf,/n_dimensions) eq 2 then psf_image=psf
endif


;
; Decide pixel subsampling rate
;
if keyword_set(image_pixel_scale) and keyword_set(psf_pixel_scale) then begin
  default_oversample=round(image_pixel_scale/psf_pixel_scale)
  if keyword_set(oversample) then if oversample ne default_oversample then message,"Pixel oversampling rates do not match!"
  decomp.oversample=default_oversample
endif else if keyword_set(oversample) then begin
  if oversample eq 1 then default_oversample=sqrt(max(n_max))/float(beta) else default_oversample=oversample
  decomp.oversample=round(default_oversample)>1
  if keyword_set(psf_image_ft) then message,/INFO,"Assuming that supplied PSF image is correctly oversampled by a factor of "+strtrim(decomp.oversample,2)+"!"
endif else begin
  decomp.oversample=1
  if keyword_set(psf_image_ft) then message,/INFO,"Assuming that supplied PSF image is not oversampled. This approach may create biases!"
endelse


;
; Reformat data and pixel weight (inverse variance) maps into a vector
;
fsize       = size(data,/DIMENSIONS)
n_pixels_x  = fsize[0]
n_pixels_y  = fsize[1]
n_pixels    = long(n_pixels_x)*long(n_pixels_y)
n_pixels_x_o= n_pixels_x*decomp.oversample
n_pixels_y_o= n_pixels_y*decomp.oversample
n_pixels_o  = n_pixels_x_o*n_pixels_y_o
data_o      = reform(rebin(data,n_pixels_x_o,n_pixels_y_o,/sample), n_pixels_o, /overwrite) ;& delvarx,data
if n_elements(noise) gt 1 then begin
  noise_o = reform(rebin(noise>0,n_pixels_x_o,n_pixels_y_o,/sample), n_pixels_o, /overwrite) ;& delvarx,noise
  masked_pixels=where(noise_o eq 0.,n_masked_pixels) & delvarx,masked_pixels
endif else begin
  noise_o=temporary(noise>0)
  n_masked_pixels=0
endelse
if max(noise_o) eq 0 then message,"All pixels masked out..."


;
; Calculate PSF convolution matrix
;
if keyword_set(psf) and not keyword_set(psf_image) then begin
  if keyword_set(overlap) then begin
    ; Force values of shapelet meta-parameters so that P_nm is square
    n_max_gamma=max(n_max)
    gamma=float(decomp.beta)*decomp.oversample
  endif
  ; Otherwise, gamma and n_max_gamma take on their default values
  P_nm=shapelets_convolution_matrix(psf,float(decomp.beta)*decomp.oversample,max(n_max),gamma,n_max_gamma)
endif else begin
  n_max_gamma=max(n_max)
  gamma=float(decomp.beta)*decomp.oversample
endelse


;
; Calculate basis functions, convolve them with the PSF and reform them into a data vector
;
if keyword_set(centre) then decomp.x=centre else decomp.x=[n_pixels_x,n_pixels_y]/2.
shapelets_make_xarr, [n_pixels_x_o, n_pixels_y_o], x1, x2, x0=decomp.x*decomp.oversample
if keyword_set(exponential) then begin
  Basis=shapelets_psi([n_max_gamma,0],x1,x2,beta=gamma,/array,$
                      ellipticity=decomp.basis_ellipticity,theta_zero=decomp.theta_zero,diamond=diamond)
endif else begin
  Basis=shapelets_phi([n_max_gamma,0],x1,x2,beta=gamma,integrate=decomp.integrate,/array,$
                      ellipticity=decomp.basis_ellipticity,theta_zero=decomp.theta_zero)
endelse
size_basis=(size(Basis,/DIMENSIONS))[2]
if keyword_set(psf_image) then begin ; This is God-awfully slow
  message,'move this to later, and check by plotting imges of each basis function that things look reasonable',/info
  for i=0,size_basis-1 do begin
    Basis[*,*,i]=convolve(Basis[*,*,i],psf_image,FT_PSF=psf_image_ft)
    print,i+1,"/",size_basis
  endfor
  MatrixT=transpose(reform(temporary(Basis),n_pixels_o,size_basis))
endif else begin
  ;MatrixT=transpose(reform(Basis,n_pixels_o,(size(Basis,/DIMENSIONS))[2]))
  size_basis=(size(Basis,/DIMENSIONS))[2]
  MatrixT=transpose(reform(temporary(Basis),n_pixels_o,size_basis))
  if keyword_set(psf) and not keyword_set(overlap) then begin
    MatrixT=transpose(P_nm)#MatrixT
  endif
endelse


;;
;; Remove basis functions to allow different n_max in two directions
;;
;if n_elements(n_max) gt 1 then begin
;  if keyword_set(polar) then begin
;    message,"Truncation scheme not available with polar shapelets. Using Cartesians instead.",/INFO,NOPRINT=silent
;    help,decomp.polar  
;  endif
;  shapelets_make_nvec, max(n_max), n1, n2, n_coeffs
;  desired_coeffs=where(n1 le n_max[0] and n2 le n_max[1])
;  MatrixT=MatrixT[desired_coeffs,*]
;endif


;
; Convert to polar shapelets, and convert complex polar shapelet basis functions into real components
;
if decomp.polar or keyword_set(diamond) or keyword_set(mushroom) then begin
  shapelets_make_nvec, max(n_max), n, m, n_coeffs, /polar
  if not keyword_set(exponential) then begin
    c2p_matrix=shapelets_polar_matrix(max(n_max),/C2P)
    if keyword_set(memory) then begin
      Matrix=complexarr((size(MatrixT,/DIMENSIONS))[1],(size(MatrixT,/DIMENSIONS))[0])
      for i=0,(size(MatrixT,/DIMENSIONS))[0]-1 do Matrix[*,i]=total(c2p_matrix[i,*])*(MatrixT[i,*])
      MatrixT=transpose((Matrix))
    endif else begin
      MatrixT=transpose(transpose(c2p_matrix) ## transpose(MatrixT))
    endelse
  endif
  m_negative=where(m lt 0,n_m_negative)
  if n_m_negative gt 0 then  MatrixT[m_negative,*]=MatrixT[m_negative,*]*complex(0,1)
  MatrixT=float(MatrixT)
endif else shapelets_make_nvec, max(n_max), n1, n2, n_coeffs


;
; Remove basis functions for various truncation schemes (not possible simultaneously, but that would be possible with more careful parsing of keywords)
;
case 1 of
  n_elements(n_max) gt 1: truncated_coeffs=where(n1 le n_max[0] and n2 le n_max[1])
  keyword_set(diamond):   truncated_coeffs=where(n+abs(m) le max(n_max))
  keyword_set(mushroom):  if mushroom gt 1 and max(n_max) ge 3 then truncated_coeffs=where(m eq 0 or n le 3) else truncated_coeffs=where(m eq 0 or n le 2)
  keyword_set(non1):      truncated_coeffs=where(n ne 1 or m ne 1)
  else:
endcase
if keyword_set(truncated_coeffs) then MatrixT=MatrixT[truncated_coeffs,*]


;
; Add extra basis functions to model the sky background
;
if keyword_set(sky) then begin
  MatrixT=[MatrixT,replicate(1.,1,n_pixels_o)]                        ; Constant
  if sky eq 2 then begin
    MatrixT=[MatrixT,reform(x1,1,n_pixels_o),reform(x2,1,n_pixels_o)] ; Plane
    sky_size=3
  endif else sky_size=1
endif else sky_size=0
delvarx,x1,x2


;
; Perform shapelet decomposition
;
if keyword_set(overlap) then begin
  ; Try to shrink the matrices if any of the pixels contain zero weight before the big matrix multiplication
  unmasked=where(noise_o ne 0,n_unmasked) ; This could be adapted to include a mask
  if n_unmasked eq n_pixels_o or n_elements(noise_o) le 1 then begin
    ; Perform "overlap integral" type of decomposition
    coeffs=MatrixT#data_o
  endif else begin
    if keyword_set(memory) then begin
      for i=0,(size(MatrixT,/DIMENSIONS))[0]-1 do decomp.coeffs[i]=total(reform(MatrixT[i,unmasked])*data_o[unmasked])
      coeffs=decomp.coeffs[0:(size(MatrixT,/DIMENSIONS))[0]-1]
    endif else coeffs=MatrixT[*,unmasked]#data_o[unmasked]
  endelse
  ; We don't know the errors on coefficients using this method, so set them to zero
  if keyword_set(polar) then begin
    coeffs_error=fltarr(n_elements(coeffs))
  endif else begin
    coeffs_error=complexarr(n_elements(coeffs))
  endelse
  Matrix=transpose(MatrixT)
endif else begin
  Matrix=transpose(MatrixT)
  ; Perform least-squares linear algebra fit
  ; (Matrix is transpose(M) of Anton p460+ or M in Lupton p84)
  if n_elements(noise_o) gt 1 then begin
    n_coeffs_left=(size(MatrixT,/DIM))[0]
    if keyword_set(memory) then begin
      for i=0L,n_pixels_o-1 do MatrixT[*,i]=MatrixT[*,i]*noise_o[i] ; slower but memory efficient version of below
    endif else MatrixT=temporary(MatrixT)*(noise_o##replicate(1,n_coeffs_left))
  endif else begin
    MatrixT=temporary(MatrixT)*noise_o ; / multiply by noise since V^-1 / since INVERSE variance
  endelse
  if keyword_set(inversion) then begin
    MatrixI=invert(MatrixT#Matrix,status) ; size of matrix to invert is only n_coeffs^2
    if status gt 0 then message,"Matrix inversion returned warning flag "+strtrim(status,2),/INFO
  endif else begin ; NB: this used to be a simple matrix inversion but can be more numerically stable using SVD (cf Berry, Hobson & Withington 2004)
    la_svd,MatrixT#Matrix,w,u,v ; Try a singular valued decomposition
    MatrixI=v ## diag_matrix(1./w*(w ge (max(w)*2e-7))) ## transpose(u)
  endelse
  ; Try to shrink the matrices if any of the pixels contain zero weight before the big matrix multiplication
  unmasked=where(noise_o ne 0,n_unmasked) ; This could be adapted to include a mask
  if n_unmasked eq n_pixels_o or n_elements(noise_o) le 1 then begin
    coeffs=MatrixI#(MatrixT#data_o)
  endif else begin
    ;message,/INFO,"Reducing size of matrix from "+strtrim(n_elements(noise_o),2)+" to "+strtrim(n_unmasked,2)
    coeffs=MatrixI#(MatrixT[*,unmasked]#data_o[unmasked])
  endelse
endelse


;
; Calculate the covariance matrix of coefficients
;
if keyword_set(overlap) then begin
  if keyword_set(cov) then begin
    coeffs_error=fltarr(n_elements(coeffs),n_elements(coeffs))
  endif else begin
    coeffs_error=fltarr(n_elements(coeffs))
  endelse
endif else begin
  if keyword_set(cov) then begin
    coeffs_error=MatrixI
  endif else begin
    coeffs_error=fltarr(n_elements(coeffs))
    for i=0,n_elements(coeffs)-1 do coeffs_error[i]=sqrt(MatrixI[i,i])
  endelse
endelse


;
; Make a recomposed image (not including sky fit but still PSF convolved)
;
if keyword_set(overlap) then begin
  recomp_o=Matrix[*,0:(size(Matrix,/DIMENSIONS))[1]-1-sky_size]#coeffs[0:n_elements(coeffs)-1-sky_size]
endif else begin
  recomp_o=Matrix#coeffs
endelse
;delvarx,Matrix
recomp=rebin(reform(float(recomp_o),n_pixels_x_o,n_pixels_y_o),n_pixels_x,n_pixels_y)


;
; Compute Chi^2 residual (directly from the original image and the recomposed
;  model, but exactly the same calculation as that done with matrices in Lupton)
;
residual=reform(data_o-recomp_o,n_pixels_x_o,n_pixels_y_o)
chisq_fudge=1.
if arg_present(wavelet_residual) then begin
  message,"Analysing residual image with wavelets",/info,noprint=silent
  ;while not keyword_set(temp_filename) do begin
  ;  random_string = "temp_"+strtrim(floor(1e6*randomu(seed)),2)
  ;  if not file_test(random_string+".fits") and not file_test(random_string+".mr") then $
  ;    temp_filename = random_string
  ;endwhile
  random_string = "\~temp_"+strtrim(floor(1e6*randomu(seed)),2)
  n_scales=1>round(alog(n_pixels_x_o>n_pixels_y_o>1)/alog(2))<10
  writefits, random_string+".fits", residual
  spawn,"mr_transform -n"+strtrim(n_scales,2)+" "+random_string+".fits "+random_string+".mr"
  wavelet_residual={data:readfits(random_string+".mr",residual_header,/SILENT),header:residual_header,n_scales:n_scales}
  file_delete,random_string+".fits",random_string+".mr",/ALLOW_NONEXISTENT
endif
if keyword_set(smooth_residual) then begin
  if smooth_residual gt 1 then begin
    message,"Smoothing residual image by a factor of "+strtrim(string(smooth_residual),2),/info,noprint=silent
    ; Reform residual  into a 2D array and add zero-padding around the edges
    n_pixels_x_p=n_pixels_x_o+2*smooth_residual
    n_pixels_y_p=n_pixels_y_o+2*smooth_residual
    residual_temp=fltarr(n_pixels_x_p,n_pixels_y_p)
    residual_temp[smooth_residual:n_pixels_x_o+smooth_residual-1,smooth_residual:n_pixels_y_o+smooth_residual-1]=reform(residual,n_pixels_x_o,n_pixels_y_o)
    residual=temporary(residual_temp)
    ; Smooth the residual image
    residual=smooth(residual,smooth_residual)
    residual=residual[smooth_residual:n_pixels_x_o+smooth_residual-1,smooth_residual:n_pixels_y_o+smooth_residual-1]
  endif
  chisq_fudge=chisq_fudge*smooth_residual
endif
if keyword_set(rebin_residual) then begin
  if rebin_residual gt 1 then begin
    message,"Rebinning residual image by a factor of "+strtrim(string(rebin_residual),2),/info,noprint=silent
    ; Reform residual  into a 2D array and add zero-padding around the edges
    n_pixels_x_p=n_pixels_x_o+rebin_residual-1-((n_pixels_x_o-1) mod rebin_residual)
    n_pixels_y_p=n_pixels_y_o+rebin_residual-1-((n_pixels_y_o-1) mod rebin_residual)
    residual_temp=fltarr(n_pixels_x_p,n_pixels_y_p)
    residual_temp[0:n_pixels_x_o-1,0:n_pixels_y_o-1]=reform(residual,n_pixels_x_o,n_pixels_y_o)
    residual=temporary(residual_temp)
    ; Rebin the residual image
    n_pixels_x_o=n_pixels_x_p/rebin_residual
    n_pixels_y_o=n_pixels_y_p/rebin_residual
    residual=rebin(residual,n_pixels_y_o,n_pixels_y_o)
  endif
  chisq_fudge=chisq_fudge*rebin_residual^3
endif
chisq=total(((reform(residual,n_pixels_o))[unmasked])^2*noise_o[unmasked])*chisq_fudge
;                    Approximate hack to get things back so ideal is ~1 -------^
;masked_pixels=where(noise eq 0.,n_masked_pixels)
dof=n_pixels-n_masked_pixels-n_elements(coeffs)
; Warn if the number of coefficients is larger than the number of pixels
if dof le 0 then begin
  message,"WARNING: No of coefficients "+strtrim(n_pixels-dof,1)+$
    " > No of pixels "+strtrim(n_pixels,1),/info,noprint=silent
  ; Could prevent least-squares from overfitting and then failing
  ; by checking this earlier? Then should set n_max to a lower number,
  ; but store n_max and recover it afterwards, setting a(^) to zero.
  dof=!values.f_infinity
endif
chisq=[chisq,chisq/float(dof)]
;if arg_present(residual) and ( n_pixels_x_o ne n_pixels_x or n_pixels_y_o ne n_pixels_y ) then $
;  residual=congrid(residual,n_pixels_x,n_pixels_y)
;plt_image,sqrt(abs(residual)),/fr,/col
;read,junk


;
; Discard sky background fit if it had been done
;
skyfit=fltarr(3)
if keyword_set(sky) then begin
  first_sky_coeff=n_elements(coeffs)-sky_size
  skyfit[0:sky_size-1]=coeffs[first_sky_coeff:*]
  coeffs=coeffs[0:first_sky_coeff-1]
  if keyword_set(cov) then message,"To do." else coeffs_error=coeffs_error[0:first_sky_coeff-1]
  if keyword_set(overlap) then skyfit[0]=skyfit[0]/n_pixels_o
endif


;;
;; Reinsert n=1 coefficients if they had been omitted
;;
;if keyword_set(non1) and min(n_max) gt 1 then begin
;  if keyword_set(cov) then message,"To do." 
;  if keyword_set(exponential) then begin
;    coeffs = [coeffs[0],0.,coeffs[1],0.,coeffs[2:*]]
;    coeffs_error = [coeffs_error[0],0.,coeffs_error[1],0.,coeffs_error[2:*]]
;  endif else begin
;    coeffs = [coeffs[0],0.,0.,coeffs[1:*]]
;    coeffs_error = [coeffs_error[0],0.,0.,coeffs_error[1:*]]
;  endelse
;endif


;
; Reinsert coefficients excluded by diamond/mushroom/non1 truncation scheme
;
if keyword_set(truncated_coeffs) then begin
  temp_coeffs=fltarr(n_coeffs)
  temp_coeffs_error=fltarr(n_coeffs)
  temp_coeffs[truncated_coeffs]=coeffs
  temp_coeffs_error[truncated_coeffs]=coeffs_error
  coeffs=temp_coeffs
  coeffs_error=temp_coeffs_error
endif


;;
;; Reinsert coefficients up to the same n_max in both directions
;;
;if n_elements(n_max) gt 1 then begin
;  temp_coeffs=fltarr(n_coeffs)
;  temp_coeffs_error=fltarr(n_coeffs)
;  temp_coeffs[desired_coeffs]=coeffs
;  if keyword_set(cov) then message,"To do." else temp_coeffs_error[desired_coeffs]=coeffs_error 
;  coeffs=temp_coeffs
;  coeffs_error=temp_coeffs_error
;endif


;
; Expand polar shapelet coefficients back into complex form
;
if decomp.polar or keyword_set(diamond) or keyword_set(mushroom) then begin
  temp_coeffs=coeffs/2
  temp_coeffs_error=coeffs_error/2
  coeffs=complex(coeffs)
  coeffs_error=complex(coeffs_error)
  for nn=1,max(n_max) do begin
    for mm=2-(nn mod 2),nn,2 do begin
      m_positive=where(n eq nn and m eq mm)
      m_negative=where(n eq nn and m eq -mm)
      coeffs[m_positive]=complex(temp_coeffs[m_positive],temp_coeffs[m_negative])
      coeffs[m_negative]=complex(temp_coeffs[m_positive],-temp_coeffs[m_negative])
      if keyword_set(cov) then message,"To do." else coeffs_error[m_positive]=complex(temp_coeffs_error[m_positive],temp_coeffs_error[m_negative])
      coeffs_error[m_negative]=complex(temp_coeffs_error[m_positive],-temp_coeffs_error[m_negative])
    endfor
  endfor
endif


;
; Convert back to Cartesian shapelet coefficients
;
if (keyword_set(diamond) or keyword_set(mushroom)) and not decomp.polar then begin
  p2c_matrix=shapelets_polar_matrix(max(n_max),/P2C)
  coeffs=p2c_matrix#coeffs
  coeffs_error=p2c_matrix#coeffs_error
endif


;
; Store results in a structure
;
decomp.coeffs=coeffs
decomp.coeffs_error=coeffs_error
decomp.n_pixels=[n_pixels_x,n_pixels_y]
decomp.chisq=chisq
decomp.sky_level=skyfit[0]
decomp.sky_slope=skyfit[1:2]


;
; Deconvolve from the PSF
;
if keyword_set(psf) and keyword_set(overlap) then begin
  if not keyword_set(inversion) then message,"TO DO: Implement SVD in matrix inversion"
  
  if decomp.polar then begin
    p2c_matrix=shapelets_polar_matrix(max(n_max),/P2C)
    coeffs=p2c_matrix#coeffs
    coeffs=invert(P_nm)#coeffs
    coeffs=c2p_matrix#coeffs
  endif else begin
    coeffs=invert(P_nm)#coeffs
  endelse
endif



;
; Convert back to polar shapelet coefficients (for nn truncation scheme and overlap deconvolution)
;
if keyword_set(polar) and not decomp.polar then shapelets_polar_convert,decomp,/POLAR



;
; Tell the world
;
return,decomp

end

