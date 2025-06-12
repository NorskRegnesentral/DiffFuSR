# 
import os   
# Now load the data from a global folder that has names as LR iamges
import tifffile
import rasterio

path_to_lr = "logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/results/worldstrat/opensrtest"
path_to_hr = "logs/blindsrsnf_aniso_worldstrat_degraded_harmfac_10000_large/version_7/results/worldstrat/diffsr/results/SR/"
if not os.path.exists(path_to_hr):
    os.makedirs(path_to_hr)
    # make subdirectories for each dataset
    os.makedirs(os.path.join(path_to_hr, "naip/geotiff"))
    os.makedirs(os.path.join(path_to_hr, "spot/geotiff"))
    os.makedirs(os.path.join(path_to_hr, "spain_crops/geotiff"))
    os.makedirs(os.path.join(path_to_hr, "spain_urban/geotiff"))
    os.makedirs(os.path.join(path_to_hr, "venus/geotiff"))


path_to_lr_naip = "load/opensrtest/100/naip/L2A"
path_to_lr_spot = "load/opensrtest/100/spot/L2A"
path_to_lr_spain_crops = "load/opensrtest/100/spain_crops/L2A"
path_to_lr_spain_urban = "load/opensrtest/100/spain_urban/L2A"
path_to_lr_venus = "load/opensrtest/100/venus/L2A"


path_to_hr_naip = "load/opensrtest/100/naip/hr"
path_to_hr_spot = "load/opensrtest/100/spot/hr"
path_to_hr_spain_crops = "load/opensrtest/100/spain_crops/hr"
path_to_hr_spain_urban = "load/opensrtest/100/spain_urban/hr"
path_to_hr_venus = "load/opensrtest/100/venus/hr"


# in Naip Name of the files in the directory only has ROI_**** in common so match the files based on this pattern
# e.g load/opensrtest/100/naip/hr/HR__ROI_00006__m_4411338_ne_12_060_20210917.tif
# e.g load/opensrtest/100/naip/L2A/ROI_00006__20210917T182021_20210917T182233_T11TQK.tif
# first making naip dictionary so store full name of HR image for each LR image
naip_dict = {}
for file_hr in os.listdir(path_to_hr_naip):
    # find the file in LR directory that has the same ROI_**** pattern
    for file_lr in os.listdir(path_to_lr_naip):
        if file_hr[4:14] == file_lr[0:10]:
            naip_dict[file_lr] = file_hr
            break

# now making spot dictionary so store full name of HR image for each LR image
spot_dict = {}
for file_hr in os.listdir(path_to_hr_spot):
    # find the file in LR directory that has the same ROI_**** pattern
    for file_lr in os.listdir(path_to_lr_spot):
        if file_hr[0:10] == file_lr[0:10]:
            spot_dict[file_lr] = file_hr
            break


# now making spain_crops dictionary so store full name of HR image for each LR image
spain_crops_dict = {}
for file_hr in os.listdir(path_to_hr_spain_crops):
    # find the file in LR directory that has the same ROI_**** pattern
    for file_lr in os.listdir(path_to_lr_spain_crops):
        if file_hr[4:14] == file_lr[4:14]:
            spain_crops_dict[file_lr] = file_hr
            break

# now making spain_urban dictionary so store full name of HR image for each LR image
spain_urban_dict = {}
for file_hr in os.listdir(path_to_hr_spain_urban):
    # find the file in LR directory that has the same ROI_**** pattern
    for file_lr in os.listdir(path_to_lr_spain_urban):
        if file_hr[4:14] == file_lr[4:14]:
            spain_urban_dict[file_lr] = file_hr
            break


# now making venus dictionary so store full name of HR image for each LR image
venus_dict = {}
for file_hr in os.listdir(path_to_hr_venus):
    # find the file in LR directory that has the same ROI_**** pattern
    for file_lr in os.listdir(path_to_lr_venus):
        if file_hr[0:8] == file_lr[0:8]:
            venus_dict[file_lr] = file_hr
            break


print("Naip Dictionary: ", naip_dict,"\n")
print("Spot Dictionary: ", spot_dict,"\n")
print("Spain Crops Dictionary: ", spain_crops_dict,"\n")
print("Spain Urban Dictionary: ", spain_urban_dict,"\n")
print("Venus Dictionary: ", venus_dict,"\n")



# load only the files that are in the lr directory of all the datasets

paths_to_files = os.listdir(path_to_lr)
# files that are in the lr directory of all the datasets, ending in .tiff
files = []
for file in paths_to_files:
    if file in naip_dict.keys() :
        #open the file 
        # instead copy the the file with a new hr name and save to the hr directory of the dataset without reading the file
        os.system("cp " + os.path.join(path_to_lr, file) + " " + os.path.join(path_to_hr, "naip/geotiff", naip_dict[file]))

    if file in spot_dict.keys() :
        #open the file 
        # instead copy the the file with a new hr name and save to the hr directory of the dataset without reading the file
        os.system("cp " + os.path.join(path_to_lr, file) + " " + os.path.join(path_to_hr, "spot/geotiff", spot_dict[file]))

    if file in spain_crops_dict.keys() :
        #open the file 
        # instead copy the the file with a new hr name and save to the hr directory of the dataset without reading the file
        os.system("cp " + os.path.join(path_to_lr, file) + " " + os.path.join(path_to_hr, "spain_crops/geotiff", spain_crops_dict[file]))

    if file in spain_urban_dict.keys() :
        #open the file 
        # instead copy the the file with a new hr name and save to the hr directory of the dataset without reading the file
        os.system("cp " + os.path.join(path_to_lr, file) + " " + os.path.join(path_to_hr, "spain_urban/geotiff", spain_urban_dict[file]))

    if file in venus_dict.keys() :
        #open the file 
        # instead copy the the file with a new hr name and save to the hr directory of the dataset without reading the file
        os.system("cp " + os.path.join(path_to_lr, file) + " " + os.path.join(path_to_hr, "venus/geotiff", venus_dict[file]))




print("Files that are in the lr directory of all the datasets: ", files)






