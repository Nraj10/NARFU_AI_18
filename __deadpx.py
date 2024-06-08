#!/usr/bin/env python3
#/home/valber/anaconda3/envs/gdal_master_env/bin/python3
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from osgeo import gdal_array
from osgeo import gdalconst
gdal.VersionInfo()

#import sys
#if len(sys.argv)>1:
#    fname=argv[1]
#else:
#    fname="/home/valber/work/edu/hackaton/img/18.Sitronics/1_20/crop_0_0_0000.tif"

import argparse

parser = argparse.ArgumentParser(description='Search for dead pixels on tiff scene')
parser.add_argument('fname', metavar='filename',
                    help='Path to file with tiff scene')
parser.add_argument('--win', default=5, type=int,
                    help='Neigbours window size (default: 5)')
parser.add_argument('--median', default=5, type=int,
                    help='Median filter window size (default: 5)')
parser.add_argument('--beta', default=5, type=float,
                    help='Threshold coefficient (default: 5)')
parser.add_argument('--minexp', default=15, type=int,
                    help='Underexpose persentage (default: 15)')
parser.add_argument('--maxexp', default=500, type=int,
                    help='Overexpose persentage (default: 500)')
parser.add_argument('--show', action='store_true',
                    help='Show scene with dead pixels marked')
parser.add_argument('--repair', action='store_true',
                    help='Repair scene')
parser.add_argument('--save', action='store_true',
                    help='Save repaired scene')

args = parser.parse_args()


win_rows, win_cols = args.win, args.win
amin, amax = args.minexp, args.maxexp
mwin=args.median//2

fname = args.fname
gdalData = gdal.Open(fname)


#print("Driver short name", gdalData.GetDriver().ShortName)
#print("Driver long name", gdalData.GetDriver().LongName)
#print("Raster size", gdalData.RasterXSize, "x", gdalData.RasterYSize)
#print("Number of bands", gdalData.RasterCount)
#print("Projection", gdalData.GetProjection())
#print("Geo transform", gdalData.GetGeoTransform())

# получаем весь растр целиком
#raster = gdalData.ReadAsArray()

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

#import time
#start_time = time.time()

band_1 = gdalData.GetRasterBand(1).ReadAsArray()
band_2 = gdalData.GetRasterBand(2).ReadAsArray()
band_3 = gdalData.GetRasterBand(3).ReadAsArray()
band_4 = gdalData.GetRasterBand(4).ReadAsArray()



#band_1_mean = ndimage.uniform_filter(band_1, (win_rows, win_cols))
#band_2_mean = ndimage.uniform_filter(band_2, (win_rows, win_cols))
#band_3_mean = ndimage.uniform_filter(band_3, (win_rows, win_cols))
#band_4_mean = ndimage.uniform_filter(band_4, (win_rows, win_cols))

#index = np.argwhere( not ((args.min/10 < band_1*10/ndimage.uniform_filter(band_1, (win_rows, win_cols))<args.max/10 ) and (args.min/10 < band_2*10/ndimage.uniform_filter(band_2, (win_rows, win_cols))<args.max/10 ) and (args.min/10 < band_3*10/ndimage.uniform_filter(band_3, (win_rows, win_cols))<args.max/10 ) and (args.min/10 < band_4*10/ndimage.uniform_filter(band_4, (win_rows, win_cols))<args.max/10 )))


#[номер строки]; [номер столбца]; [номер канала]; [«битое» значение]; [исправленное значение]


index1 = np.argwhere((ndimage.uniform_filter(band_1, (win_rows, win_cols))/10*amin/10>band_1) | (ndimage.uniform_filter(band_1, (win_rows, win_cols))/100*amax<band_1))

index2 = np.argwhere((ndimage.uniform_filter(band_2, (win_rows, win_cols))/10*amin/10>band_2) | (ndimage.uniform_filter(band_2, (win_rows, win_cols))/100*amax<band_2))   
    
index3 = np.argwhere((ndimage.uniform_filter(band_3, (win_rows, win_cols))/10*amin/10>band_3) | (ndimage.uniform_filter(band_3, (win_rows, win_cols))/100*amax<band_3))

index4 = np.argwhere((ndimage.uniform_filter(band_4, (win_rows, win_cols))/10*amin/10>band_4) | (ndimage.uniform_filter(band_4, (win_rows, win_cols))/100*amax<band_4))

#print("--- %s seconds ---" % (time.time() - start_time))

for (row,col) in index1:
    print("{}; {}; {}; {}; {}".format(row,col,1,band_1[row,col],np.median(band_1[row-mwin:row+mwin,col-mwin:col+mwin]).astype(int)))
for (row,col) in index2:
    print("{}; {}; {}; {}; {}".format(row,col,2,band_2[row,col],np.median(band_2[row-mwin:row+mwin,col-mwin:col+mwin]).astype(int)))
for (row,col) in index3:
    print("{}; {}; {}; {}; {}".format(row,col,3,band_3[row,col],np.median(band_3[row-mwin:row+mwin,col-mwin:col+mwin]).astype(int)))
for (row,col) in index4:
    print("{}; {}; {}; {}; {}".format(row,col,4,band_4[row,col],np.median(band_4[row-mwin:row+mwin,col-mwin:col+mwin]).astype(int)))
    

if args.show and not args.repair:
    
    fig = plt.figure(figsize=(10, 10))
    '''
    mn=band_1.min()
    mx=band_1.max()
    r=(band_1-mn)/(mx-mn)
    mn=band_2.min()
    mx=band_2.max()
    g=(band_2-mn)/(mx-mn)
    mn=band_3.min()
    mx=band_3.max()
    b=(band_3-mn)/(mx-mn)
    npa=np.stack([r,g,b], axis=2)
    plt.imshow(npa)
    '''
    mn=np.mean([band_1,band_2,band_3])+2*np.std([band_1,band_2,band_3])
    plt.imshow(np.stack([band_1,band_2,band_3], axis=2).astype(float)/mn)
    ax = plt.gca()
                
    for i, val in enumerate(index4):
        ax.add_patch(plt.Circle((val[1], val[0]), 2.1, color='k', fill=False))
    for i, val in enumerate(index3):
        ax.add_patch(plt.Circle((val[1], val[0]), 2, color='b', fill=False))
    for i, val in enumerate(index2):
        ax.add_patch(plt.Circle((val[1], val[0]), 1.9, color='g', fill=False))
    for i, val in enumerate(index1):
        ax.add_patch(plt.Circle((val[1], val[0]), 1.8, color='r', fill=False))


 #   start_time = time.time()
    
    band_1=np.asfarray(band_1)
    band_2=np.asfarray(band_2)
    band_3=np.asfarray(band_3)
    
    r_mean = ndimage.uniform_filter(band_1, (win_rows, win_cols))
    r_sqr_mean = ndimage.uniform_filter(band_1**2, (win_rows, win_cols))
    r_std = np.sqrt(r_sqr_mean - r_mean**2)
    
    #should be cheaper way, but no:
    #r_mean = ndimage.uniform_filter(band_1, (win_rows, win_cols))
    #r_std = np.sqrt(ndimage.uniform_filter((band_1-r_mean)**2, (win_rows, win_cols))) 

    g_mean = ndimage.uniform_filter(band_2, (win_rows, win_cols))
    g_sqr_mean = ndimage.uniform_filter(band_2**2, (win_rows, win_cols))
    g_std = np.sqrt(g_sqr_mean - g_mean**2)

    b_mean = ndimage.uniform_filter(band_3, (win_rows, win_cols))
    b_sqr_mean = ndimage.uniform_filter(band_3**2, (win_rows, win_cols))
    b_std = np.sqrt(b_sqr_mean - b_mean**2)

    #r_var_mean = ndimage.uniform_filter(r_var, (win_rows, win_cols))
    #g_var_mean = ndimage.uniform_filter(g_var, (win_rows, win_cols))
    #b_var_mean = ndimage.uniform_filter(b_var, (win_rows, win_cols))

    r_diff_mean = ndimage.uniform_filter(np.abs(r_mean-band_1), (win_rows, win_cols))
    g_diff_mean = ndimage.uniform_filter(np.abs(g_mean-band_2), (win_rows, win_cols))
    b_diff_mean = ndimage.uniform_filter(np.abs(b_mean-band_3), (win_rows, win_cols))

    beta=args.beta
    index = np.argwhere(np.abs(r_mean-band_1)+np.abs(g_mean-band_2)+np.abs(b_mean-band_3) > beta*(r_diff_mean+g_diff_mean+b_diff_mean))


#    print("--- %s seconds ---" % (time.time() - start_time))
    
    for i, val in enumerate(index):
        ax.add_patch(plt.Circle((val[1], val[0]), 2.2, color='m', fill=False))

#    start_time = time.time()    

    index = np.argwhere(np.abs(r_mean-band_1)+np.abs(g_mean-band_2)+np.abs(b_mean-band_3) > 3*(r_std+g_std+b_std))

#    print("--- %s seconds ---" % (time.time() - start_time))

    for i, val in enumerate(index):
        ax.add_patch(plt.Circle((val[1], val[0]), 2.3, color='c', fill=False))    

        
    if args.repair:
        plt.show(block=False)
    else:
        plt.show()


if args.repair:

    for (row,col) in index1:
        band_1[row,col]=np.median(band_1[row-mwin:row+mwin,col-mwin:col+mwin]).astype(int)
    for (row,col) in index2:
        band_2[row,col]=np.median(band_2[row-mwin:row+mwin,col-mwin:col+mwin]).astype(int)
    for (row,col) in index3:
        band_3[row,col]=np.median(band_3[row-mwin:row+mwin,col-mwin:col+mwin]).astype(int)
    for (row,col) in index4:
        band_4[row,col]=np.median(band_4[row-mwin:row+mwin,col-mwin:col+mwin]).astype(int)
    
    if args.show:
        fig = plt.figure(figsize=(10, 10))
        '''
        mn=band_1.min()
        mx=band_1.max()
        r=(band_1-mn)/(mx-mn)
        mn=band_2.min()
        mx=band_2.max()
        g=(band_2-mn)/(mx-mn)
        mn=band_3.min()
        mx=band_3.max()
        b=(band_3-mn)/(mx-mn)    
        npa=np.stack([r,g,b], axis=2)
        plt.imshow(npa)
        '''
        mn=np.mean([band_1,band_2,band_3])+2*np.std([band_1,band_2,band_3])
        plt.imshow(np.stack([band_1,band_2,band_3], axis=2).astype(float)/mn)
        ax = plt.gca()

        for i, val in enumerate(index4):
            ax.add_patch(plt.Circle((val[1], val[0]), 2, color='k', fill=False))
        for i, val in enumerate(index3):
            ax.add_patch(plt.Circle((val[1], val[0]), 2, color='b', fill=False))
        for i, val in enumerate(index2):
            ax.add_patch(plt.Circle((val[1], val[0]), 2, color='g', fill=False))
        for i, val in enumerate(index1):
            ax.add_patch(plt.Circle((val[1], val[0]), 2, color='r', fill=False))
    
        plt.show()

    if args.save:
        projection = gdalData.GetProjection()
        transform = gdalData.GetGeoTransform()
        xsize = gdalData.RasterXSize
        ysize = gdalData.RasterYSize
        gdalData = None

        format = "GTiff"
        driver = gdal.GetDriverByName( format )
        metadata = driver.GetMetadata()
        if  metadata[ gdal.DCAP_CREATE ] == "YES":
            outRaster = driver.Create( fname+'.repaired', xsize, ysize, 4, gdal.GDT_UInt16 )
            outRaster.SetProjection( projection )
            outRaster.SetGeoTransform( transform )
            outRaster.GetRasterBand( 1 ).WriteArray( band_1 )
            outRaster.GetRasterBand( 2 ).WriteArray( band_2 )
            outRaster.GetRasterBand( 3 ).WriteArray( band_3 )
            outRaster.GetRasterBand( 4 ).WriteArray( band_4 )
            outRaster = None
        else:
            print("Driver %s does not support Create() method." % format)
            
