import autolens as al

arr = al.Array2D.from_fits(file_path="hlsp_clash_hst_acs-30mas_macs1206_f814w_v1_drz.fits", pixel_scales=0.1, hdu=0)
print(arr.shape_native)
arr1 = al.Array2D.from_fits(file_path="acs00.fits", pixel_scales=0.1, hdu=0)
print(arr1.shape_native)

