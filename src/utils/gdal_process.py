"""
from https://github.com/langnico/gdal_processing/blob/main/gdal_process.py
"""
import os
import osgeo
from osgeo import gdal, osr, ogr, gdalconst
import numpy as np
from skimage.transform import resize
import progressbar as pbar
from zipfile import ZipFile
from glob import glob
import boto3
import time

gdal.UseExceptions()


# To read sentinel-2 from AWS s3, load and set the temporary IAM role credentials.
if 'ROLE_ARN' in os.environ and os.environ['ROLE_ARN'] not in ['None', None]:
    print('setting GDAL configs to temporary role credentials...')
    # create an STS client object that represents a live connection to the STS service
    sts_client = boto3.client('sts')

    # Call the assume_role method of the STSConnection object and pass the role ARN and a role session name.
    assumed_role_object=sts_client.assume_role(
        RoleArn=os.environ['ROLE_ARN'],
        RoleSessionName="AssumeRoleSession1"
    )

    # From the response that contains the assumed role, get the temporary
    # credentials that can be used to make subsequent API calls
    credentials=assumed_role_object['Credentials']

    # Use the temporary credentials that AssumeRole returns to configure GDAL
    gdal.SetConfigOption('AWS_ACCESS_KEY_ID', credentials['AccessKeyId'])
    gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', credentials['SecretAccessKey'])
    gdal.SetConfigOption('AWS_SESSION_TOKEN', credentials['SessionToken'])
    gdal.SetConfigOption('AWS_REQUEST_PAYER', 'requester')


# Get AWS credentials from local env variables
#try:
#    print("Setting gdal AWS credentials from ENV VARS")
#    gdal.SetConfigOption('AWS_ACCESS_KEY_ID', os.environ['AWS_ACCESS_KEY_ID'])
#    gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY', os.environ['AWS_SECRET_ACCESS_KEY'])
#    gdal.SetConfigOption('AWS_REQUEST_PAYER', os.environ['AWS_REQUEST_PAYER'])

#except KeyError:
#    print("You have to set set the environment variables before (e.g. 'source /path/to/.aws_configs') \n \
#    The file .aws_configs should contain the following: \n \
#        export AWS_ACCESS_KEY_ID=PUT_YOUR_KEY_ID_HERE \n \
#        export AWS_SECRET_ACCESS_KEY=PUT_YOUR_SECRET_ACCESS_KEY_HERE \n \
#        export AWS_REQUEST_PAYER=requester")

GDAL_TYPE_LOOKUP = {'float32': gdal.GDT_Float32,
                    'float64': gdal.GDT_Float64,
                    'uint16': gdal.GDT_UInt16,
                    'uint8': gdal.GDT_Byte}

RESAMPLE_ALGO_DICT = {'no': None,
                     'max': gdalconst.GRA_Max,
                      'mean': gdalconst.GRA_Average,
                      'bilinear': gdalconst.GRA_Bilinear,
                      'cubic': gdalconst.GRA_Cubic,
                      'nearest': gdalconst.GRA_NearestNeighbour}


def sort_band_arrays(band_arrays, channels_last=True):
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    out_arr = []
    for b in bands:
        out_arr.append(band_arrays[b])
    out_arr = np.array(out_arr)
    if channels_last:
        out_arr = np.moveaxis(out_arr, source=0, destination=-1)
    return out_arr


def get_points_array_from_shape(shape_path, value_field='Elevation'):
    Shapefile = ogr.Open(shape_path)
    Shapefile_layer = Shapefile.GetLayer()

    n_points = Shapefile_layer.GetFeatureCount()
    points = np.empty((n_points, 3), dtype=np.double)
    print('number of features in shapefile: ', n_points)
    for i in range(n_points):
        feat = Shapefile_layer.GetNextFeature()
        # geom = feat.GetGeometryRef()
        points[i] = [feat.GetField('Longitude'), feat.GetField('Latitude'), feat.GetField(value_field)]
    filename = shape_path.replace(".shp", ".npy")
    np.save(filename, points)
    return points


def rasterize_points(points, refDataset, lims=None, scale=1, average=True, interpolate=True, keep_last=False, out_type=np.float32, nodata_value=np.nan):
    if lims is not None:
        lims = [i * scale for i in lims]
        xmin, ymin, xmax, ymax = lims
    else:
        xmin, ymin, xmax, ymax = 0, 0, refDataset.RasterXSize - 1, refDataset.RasterYSize - 1

    length_x = xmax - xmin + 1
    length_y = ymax - ymin + 1

    print('length_y: ', length_y, type(length_y))
    print('length_x: ', length_x, type(length_x))
    mask_value = np.zeros((length_y, length_x), dtype=out_type)
    mask_count = np.zeros((length_y, length_x), dtype=np.float32)

    xoff, a, b, yoff, d, e = refDataset.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(refDataset.GetProjection())
    srsLatLon = osr.SpatialReference()
    srsLatLon.SetWellKnownGeogCS("WGS84")
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        srsLatLon.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(srsLatLon, srs)

    # change pixel size
    a = a / scale
    e = e / scale

    for i in points:
        # get z value
        value = i[2].astype(out_type)

        (xp, yp, h) = ct.TransformPoint(i[0], i[1], 0.)
        xp -= xoff
        yp -= yoff
        # matrix inversion
        det_inv = 1. / (a * e - d * b)
        x = (e * xp - b * yp) * det_inv
        y = (-d * xp + a * yp) * det_inv
        x, y = (int(x), int(y))
        # x,y = to_xy(i[0], i[1], refDataset)
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax:
            x1 = x - xmin
            y1 = y - ymin

            mask_count[y1, x1] += 1
            if keep_last:
                mask_value[y1, x1] = value
            else:
                # sum up
                mask_value[y1, x1] += value

    # Elementwise average with mask_value / mask_count
    empty_indices = mask_count == 0
    if average:
        mask_value = np.divide(mask_value, mask_count, where=~empty_indices)

    if scale > 1:
        # sigma = scale
        sigma = scale / np.pi
        mask_value = ndimage.gaussian_filter(mask_value.astype(np.float32), sigma=sigma)

        mask_value = block_reduce(mask_value, (scale, scale), np.mean)
        # mask = mask[::scale,::scale]
        print(
            'GT points were smoothed on High resolution with a Gaussian \sigma = {:.2f} and downsampled {} times'.format(
                sigma, scale))

    # set empty values to nan
    mask_value[empty_indices] = nodata_value
    print('dtype mask_value:', mask_value.dtype)

    if interpolate:
        # Interpolate enclosed nan values
        mask_value = interpolate_enclosed_nans(mask_value, empty_mask_2d=empty_indices)
        print('dtype mask_value interp:', mask_value.dtype)
        print('value at top left [0,0]', mask_value[0, 0])
        print('nb. nan values', np.sum(np.isnan(mask_value)))

    return mask_value


def read_tree_biomass(refDataset, lims_ref_pixel):
    # individual tree biomass should be accumulated not averaged
    biomass_raster = rasterize_points(
        Input='/scratch2/data/GEDI/Gabon/biomass_field/Gabon_Mondah/mondah_trees_agb_WGS84_clean.shp',
        refDataset=refDataset, lims=lims_ref_pixel, scale=1, average=False, value_field='m.agb', interpolate=False)

    np.save('/scratch2/tmp/biomass_mondah_raster.npy', biomass_raster)


def get_tile_info(refDataset):
    tile_info = {}
    tile_info['projection'] = refDataset.GetProjection()
    tile_info['geotransform'] = refDataset.GetGeoTransform()
    tile_info['width'] = refDataset.RasterXSize
    tile_info['height'] = refDataset.RasterYSize
    return tile_info


def save_array_as_geotif(out_path, array, tile_info, out_type=None, out_bands=1, dstnodata=None,
                         compress='DEFLATE', predictor=2):
    if out_type is None:
        out_type = array.dtype.name
    out_type = GDAL_TYPE_LOOKUP[out_type]
    # PACKBITS is a lossless compression.
    # predictor=2 saves horizontal differences to previous value (useful for empty regions)
    dst_ds = gdal.GetDriverByName('GTiff').Create(out_path, tile_info['width'], tile_info['height'], out_bands, out_type,
                                                  options=['COMPRESS={}'.format(compress), 'PREDICTOR={}'.format(predictor)])
    dst_ds.SetGeoTransform(tile_info['geotransform'])
    dst_ds.SetProjection(tile_info['projection'])
    dst_ds.GetRasterBand(1).WriteArray(array)  # write r-band to the raster
    if dstnodata is not None:
        dst_ds.GetRasterBand(1).SetNoDataValue(dstnodata)
    dst_ds.FlushCache()  # write to disk
    dst_ds = None


def get_points_with_patches_in_tile(points_dict, patch_size, refDataset):
    invalid_point_ids = []
    height = refDataset.RasterYSize
    width = refDataset.RasterXSize

    # to check if the patch center is within a margin (patch is larger when it is aligned to 60m pixels. e.g. 24 istead of 15 pixels)
    patch_size_align = int((patch_size + 6) // 6 + 1) * 6

    count_skipped = 0
    for p_id in list(points_dict.keys()):
        x_center, y_center = to_xy(lon=points_dict[p_id]['lon'], lat=points_dict[p_id]['lat'], ds=refDataset)

        x_topleft = x_center - patch_size // 2
        y_topleft = y_center - patch_size // 2
        x_topleft_align = int(x_topleft // 6) * 6
        y_topleft_align = int(y_topleft // 6) * 6

        # check if extended patch extraction is possible
        is_valid = (0 <= x_topleft_align
                    and (x_topleft_align + patch_size_align) < width
                    and 0 <= y_topleft_align
                    and (y_topleft_align + patch_size_align) < height)
        if not is_valid:
            invalid_point_ids.append(p_id)
            del points_dict[p_id]
            count_skipped += 1
        else:
            points_dict[p_id]['x_topleft'] = x_topleft
            points_dict[p_id]['y_topleft'] = y_topleft

    if count_skipped > 0:
        print('{} locations skipped: patch center is too close to border'.format(count_skipped))
    return points_dict, invalid_point_ids


def extract_patches_from_label_mask(label_mask, points_dict, patch_size):
    label_patches = {}

    for p_id in points_dict:
        x_topleft = points_dict[p_id]['x_topleft']
        y_topleft = points_dict[p_id]['y_topleft']

        label_patches[p_id] = label_mask[y_topleft:y_topleft + patch_size,
                                         x_topleft:x_topleft + patch_size]

    return label_patches


def to_xy(lon, lat, ds):
    xoff, a, b, yoff, d, e = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    srsLatLon = osr.SpatialReference()
    srsLatLon.SetWellKnownGeogCS("WGS84")
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        srsLatLon.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
    ct = osr.CoordinateTransformation(srsLatLon, srs)

    (xp, yp, h) = ct.TransformPoint(lon, lat, 0.)
    xp -= xoff
    yp -= yoff
    # matrix inversion
    det_inv = 1. / (a * e - d * b)
    x = (e * xp - b * yp) * det_inv
    y = (-d * xp + a * yp) * det_inv
    return (int(x), int(y))


def to_latlon(x, y, ds):
    bag_gtrn = ds.GetGeoTransform()
    bag_proj = ds.GetProjectionRef()
    bag_srs = osr.SpatialReference(bag_proj)
    geo_srs = bag_srs.CloneGeogCS()
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        bag_srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
        geo_srs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(bag_srs, geo_srs)

    # in a north up image:
    originX = bag_gtrn[0]
    originY = bag_gtrn[3]
    pixelWidth = bag_gtrn[1]
    pixelHeight = bag_gtrn[5]

    easting = originX + pixelWidth * x + bag_gtrn[2] * y
    northing = originY + bag_gtrn[4] * x + pixelHeight * y

    geo_pt = transform.TransformPoint(easting, northing)[:2]
    lon = geo_pt[0]
    lat = geo_pt[1]
    return lat, lon


def get_lat_lon_per_pixel(xoff, yoff, height, width, ds):
    y_col = np.expand_dims(np.arange(0, height), axis=1)
    x_row = np.expand_dims(np.arange(0, width), axis=0)

    # 2d arrays (add the top left xy coordinate w.r.t to the original tile)
    y_coords = np.repeat(y_col, repeats=width, axis=1) + yoff
    x_coords = np.repeat(x_row, repeats=height, axis=0) + xoff

    # compute lat, lon for each pixel coordinate
    to_latlon_vec = np.vectorize(to_latlon)
    lat, lon = to_latlon_vec(x_coords, y_coords, ds)

    return lat, lon


def get_lat_lon_patches_slow(points_dict, refDataset, patch_size):
    latlon_patches = {}

    for p_id in points_dict:
        latlon_patches[p_id] = {}
        p = points_dict[p_id]
        #print('DEBUG:', p)
        lat, lon = get_lat_lon_per_pixel(xoff=p['x_topleft'], yoff=p['y_topleft'],
                                         height=patch_size, width=patch_size, ds=refDataset)
        latlon_patches[p_id]['lat'] = lat
        latlon_patches[p_id]['lon'] = lon
    return latlon_patches


def create_latlon_mask(height, width, refDataset, out_type=np.float32):
    # compute lat, lon of top-left and bottom-right corners
    lat_topleft, lon_topleft = to_latlon(x=0, y=0, ds=refDataset)
    lat_bottomright, lon_bottomright = to_latlon(x=width-1, y=height-1, ds=refDataset)

    # interpolate between the corners
    lat_col = np.linspace(start=lat_topleft, stop=lat_bottomright, num=height).astype(out_type)
    lon_row = np.linspace(start=lon_topleft, stop=lon_bottomright, num=width).astype(out_type)

    # expand dimensions of row and col vector to repeat
    lat_col = lat_col[:, None]
    lon_row = lon_row[None, :]

    # repeat column and row to get 2d arrays --> lat lon coordinate for every pixel
    lat_mask = np.repeat(lat_col, repeats=width, axis=1)
    lon_mask = np.repeat(lon_row, repeats=height, axis=0)

    print('lat_mask.shape: ', lat_mask.shape)
    print('lon_mask.shape: ', lon_mask.shape)

    return lat_mask, lon_mask


def get_lat_lon_patches(points_dict, refDataset, patch_size):
    latlon_patches = {}

    # create masks for full image
    lat_mask, lon_mask = create_latlon_mask(height=refDataset.RasterYSize, width=refDataset.RasterXSize,
                                            refDataset=refDataset)

    for p_id in points_dict:
        latlon_patches[p_id] = {}
        p = points_dict[p_id]
        #print('DEBUG:', p)
        lat_patch = lat_mask[p['y_topleft']:p['y_topleft'] + patch_size,
                             p['x_topleft']:p['x_topleft'] + patch_size]

        lon_patch = lon_mask[p['y_topleft']:p['y_topleft'] + patch_size,
                             p['x_topleft']:p['x_topleft'] + patch_size]

        latlon_patches[p_id]['lat'] = lat_patch
        latlon_patches[p_id]['lon'] = lon_patch
    return latlon_patches


def get_patches_from_tif(points_dict, patch_size, path_tif):
    ds = gdal.Open(path_tif)
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray()
    patches = {}
    for p_id in points_dict:
        patches[p_id] = {}
        p = points_dict[p_id]
        patches[p_id] = array[p['y_topleft']:p['y_topleft'] + patch_size,
                              p['x_topleft']:p['x_topleft'] + patch_size]
    return patches


def read_patch_from_vsis3(xoff, yoff, win_xsize, win_ysize, ds):
    # get the first raster band
    band = ds.GetRasterBand(1)
    return band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize)


# for a single patch location (lat, lon)
def read_sentinel2_patch_from_vsis3_v1(lon, lat, patch_size, data_path, bucket='sentinel-s2-l2a'):
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    resolutions = [60, 10, 10, 10, 20, 20, 20, 10, 20, 60, 20, 20]

    target_shape = (patch_size, patch_size)

    bands10m = ['B02', 'B03', 'B04', 'B08']
    bands20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    bands60m = ['B01', 'B09']  # 'B10' is missing in 2A, exists only in 1C

    bands_dir = {10: {'band_names': bands10m, 'subdir': 'R10m', 'scale': 1},
                 20: {'band_names': bands20m, 'subdir': 'R20m', 'scale': 2},
                 60: {'band_names': bands60m, 'subdir': 'R60m', 'scale': 6}}

    # get 10m xoff, yoff
    B2_path = os.path.join('/vsis3', bucket, data_path, 'R10m', 'B02.jp2')
    print('B2_path: ', B2_path)
    ds = gdal.Open(B2_path)
    x_center, y_center = to_xy(lon, lat, ds)
    print('x_center, y_center : ', x_center, y_center)

    band_arrays = {}

    for res in bands_dir.keys():
        bands_dir[res]['band_arrays'] = np.zeros(shape=(patch_size, patch_size, len(bands_dir[res]['band_names'])))
        for i in range(len(bands_dir[res]['band_names'])):
            band_name = bands_dir[res]['band_names'][i]
            path_band = os.path.join('/vsis3', bucket, data_path, bands_dir[res]['subdir'], band_name + '.jp2')
            print('path_band: ', path_band)
            scale = bands_dir[res]['scale']
            ds = gdal.Open(path_band)
            patch = read_patch_from_vsis3(xoff=x_center // scale - patch_size // scale // 2,
                                          yoff=y_center // scale - patch_size // scale // 2,
                                          win_xsize=patch_size // scale,
                                          win_ysize=patch_size // scale,
                                          ds=ds)
            print(patch.shape)
            if patch.shape != target_shape:
                patch = resize(patch, target_shape, mode='reflect', order=3)  # bicubic
            band_arrays[band_name] = patch

    return band_arrays


def enlarge_to_60pixel(xmin, xmax):
    xmin = int(xmin // 6) * 6
    xmax = int((xmax + 1) // 6) * 6 - 1
    return xmin, xmax


def get_patch(array, xoff, yoff, win_xsize, win_ysize):
    return array[yoff:yoff + win_ysize, xoff:xoff + win_xsize]


# read multiple patch locations (lat, lon)
def read_sentinel2_patches_from_vsis3(points_dict, patch_size, data_path, bucket='sentinel-s2-l2a', maxCount=None, read_full_arrays=True):

    patch_size_align = int((patch_size + 6) // 6 + 1) * 6

    bands10m = ['B02', 'B03', 'B04', 'B08']
    bands20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    bands60m = ['B01', 'B09']  # 'B10' is missing in 2A, exists only in 1C

    bands_dir = {10: {'band_names': bands10m, 'subdir': 'R10m', 'scale': 1},
                 20: {'band_names': bands20m, 'subdir': 'R20m', 'scale': 2},
                 60: {'band_names': bands60m, 'subdir': 'R60m', 'scale': 6}}

    # gdal open all bands
    print('Opening bands with gdal vsis3...')

    for res in bands_dir.keys():
        bands_dir[res]['ds_list'] = []
        if read_full_arrays:
            bands_dir[res]['band_data_list'] = []
        for i in range(len(bands_dir[res]['band_names'])):
            band_name = bands_dir[res]['band_names'][i]
            path_band = os.path.join('/vsis3', bucket, data_path, bands_dir[res]['subdir'], band_name + '.jp2')

            # # debug test vsizip gabon gedi simulated 1700 points 49 sec
            # path_band = os.path.join('/vsizip//scratch2/tmp/dummy_patches/32MQE_REFds/S2A_MSIL2A_20190529T092031_N0212_R093_T32MQE_20190529T134134.zip/S2A_MSIL2A_20190529T092031_N0212_R093_T32MQE_20190529T134134.SAFE/GRANULE/L2A_T32MQE_A020539_20190529T093703/IMG_DATA/',
            #                          bands_dir[res]['subdir'], 'T32MQE_20190529T092031_' + band_name + '_{}m.jp2'.format(bands_dir[res]['scale']*10))
            #
            # # debug test vsizip CH biomass 1427 patches: 117 sec (read_full_array: 56 sec)
            # path_band = os.path.join(
            #     '/vsizip//scratch2/tmp/dummy_patches/32TMT_REFds/S2A_MSIL2A_20190530T103031_N0212_R108_T32TMT_20190530T132605.zip/S2A_MSIL2A_20190530T103031_N0212_R108_T32TMT_20190530T132605.SAFE/GRANULE/L2A_T32TMT_A020554_20190530T103655/IMG_DATA/',
            #     bands_dir[res]['subdir'],
            #     'T32TMT_20190530T103031_' + band_name + '_{}m.jp2'.format(bands_dir[res]['scale'] * 10))
            #
            # # debug test unzipped CH biomass 1427 patches: 116 sec (read_full_array: 58 sec)
            # path_band = os.path.join(
            #     '/scratch2/tmp/dummy_patches/32TMT_REFds/S2A_MSIL2A_20190530T103031_N0212_R108_T32TMT_20190530T132605.SAFE/GRANULE/L2A_T32TMT_A020554_20190530T103655/IMG_DATA/',
            #     bands_dir[res]['subdir'],
            #     'T32TMT_20190530T103031_' + band_name + '_{}m.jp2'.format(bands_dir[res]['scale'] * 10))

            print('path_band: ', path_band)
            ds = gdal.Open(path_band)
            bands_dir[res]['ds_list'].append(ds)

            if read_full_arrays:
                # read all band data to memory once
                print('reading full band array...')
                band = ds.GetRasterBand(1)
                band_data = band.ReadAsArray()
                bands_dir[res]['band_data_list'].append(band_data)

    # loop through patch location
    band_arrays = {}
    count = 0
    print('reading patches...')
    for p_id in pbar.progressbar(points_dict):
        count += 1

        # get 10m xoff, yoff
        x_center, y_center = to_xy(lon=points_dict[p_id]['lon'], lat=points_dict[p_id]['lat'],
                                   ds=bands_dir[10]['ds_list'][0])
        # print('10m res. x_center, y_center : ', x_center, y_center )
        x_topleft = x_center - patch_size // 2
        y_topleft = y_center - patch_size // 2
        # print('x_topleft, y_topleft            ', x_topleft, y_topleft)
        # print('patch_size', patch_size)
        # round 10m xoff, yoff, patch_size to align with 60m resolution
        x_topleft_align = int(x_topleft // 6) * 6
        y_topleft_align = int(y_topleft // 6) * 6

        # top left in the cropped patch
        y_patch_topleft = y_topleft - y_topleft_align
        x_patch_topleft = x_topleft - x_topleft_align

        target_shape = (patch_size_align, patch_size_align)

        # print('x_topleft_align, y_topleft_align', x_topleft_align, y_topleft_align)
        # print('patch_size_align', patch_size_align)
        # print('y_bottomright_align - y_topleft_align: ', y_bottomright_align - y_topleft_align)
        # print('x_bottomright_align - x_topleft_align: ', x_bottomright_align - x_topleft_align)

        # init array dict for point id
        band_arrays[p_id] = {}
        band_arrays[p_id]['x_topleft'] = x_topleft
        band_arrays[p_id]['y_topleft'] = y_topleft
        for res in bands_dir.keys():
            scale = bands_dir[res]['scale']
            for i in range(len(bands_dir[res]['band_names'])):
                band_name = bands_dir[res]['band_names'][i]
                if read_full_arrays:
                    patch = get_patch(array=bands_dir[res]['band_data_list'][i],
                                      xoff=x_topleft_align // scale,
                                      yoff=y_topleft_align // scale,
                                      win_xsize=patch_size_align // scale,
                                      win_ysize=patch_size_align // scale)
                else:
                    ds = bands_dir[res]['ds_list'][i]
                    patch = read_patch_from_vsis3(xoff=x_topleft_align // scale,
                                                  yoff=y_topleft_align // scale,
                                                  win_xsize=patch_size_align // scale,
                                                  win_ysize=patch_size_align // scale,
                                                  ds=ds)
                # print('patch temp shape: ', patch.shape)
                if patch.shape != target_shape:
                    patch = resize(patch, target_shape, mode='reflect', order=3)  # bicubic
                    # print('patch temp resized shape: ', patch.shape)

                # get final patch (with centered gt pixel)
                patch_final = patch[y_patch_topleft:y_patch_topleft + patch_size,
                                    x_patch_topleft:x_patch_topleft + patch_size]

                # print('patch final shape: ', patch_final.shape)
                band_arrays[p_id][band_name] = patch_final

        if maxCount is not None:
            if count == maxCount:
                print('reached maxCount')
                break

    return band_arrays


def get_shape_attributes(path_file, do_print=False):
    source = ogr.Open(path_file)
    layer = source.GetLayer()
    attributes = []
    ldefn = layer.GetLayerDefn()
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        attributes.append(fdefn.name)
    if do_print:
        print(attributes)
    return attributes


# read multiple patch locations (lat, lon)
def read_sentinel2_patches(points_dict, patch_size, data_path, band_prefix=None, from_aws=False, bucket='sentinel-s2-l2a', maxCount=None, read_full_arrays=True):

    patch_size_align = int((patch_size + 6) // 6 + 1) * 6

    bands10m = ['B02', 'B03', 'B04', 'B08']
    bands20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL']
    bands60m = ['B01', 'B09']  # 'B10' is missing in 2A, exists only in 1C

    bands_dir = {10: {'band_names': bands10m, 'subdir': 'R10m', 'scale': 1},
                 20: {'band_names': bands20m, 'subdir': 'R20m', 'scale': 2},
                 60: {'band_names': bands60m, 'subdir': 'R60m', 'scale': 6}}

    # gdal open all bands
    print('Opening bands with gdal vsis3...')

    for res in bands_dir.keys():
        bands_dir[res]['ds_list'] = []
        if read_full_arrays:
            bands_dir[res]['band_data_list'] = []
        for i in range(len(bands_dir[res]['band_names'])):
            band_name = bands_dir[res]['band_names'][i]

            if from_aws:
                path_band = os.path.join('/vsis3', bucket, data_path, bands_dir[res]['subdir'], band_name + '.jp2')
            else:
                path_band = os.path.join('/vsizip/' + data_path, bands_dir[res]['subdir'],
                                         '{}_{}_{}m.jp2'.format(band_prefix, band_name, bands_dir[res]['scale']*10))

            print('path_band: ', path_band)
            ds = gdal.Open(path_band)
            bands_dir[res]['ds_list'].append(ds)

            if read_full_arrays:
                # read all band data to memory once
                print('reading full band array...')
                band = ds.GetRasterBand(1)
                band_data = band.ReadAsArray()
                bands_dir[res]['band_data_list'].append(band_data)

    print("Opening CLD band...")
    path_band = os.path.join('/vsizip/' + data_path.replace("IMG_DATA", "QI_DATA"), 'MSK_CLDPRB_20m.jp2')
    bands20m.append('CLD')
    ds = gdal.Open(path_band)
    bands_dir[20]['ds_list'].append(ds)
    if read_full_arrays:
        # read all band data to memory once
        print('reading full band array...')
        band = ds.GetRasterBand(1)
        band_data = band.ReadAsArray()
        bands_dir[20]['band_data_list'].append(band_data)

    # loop through patch location
    band_arrays = {}
    count = 0
    print('reading patches...')
    for p_id in pbar.progressbar(points_dict):
        count += 1

        # get 10m xoff, yoff
        x_center, y_center = to_xy(lon=points_dict[p_id]['lon'], lat=points_dict[p_id]['lat'],
                                   ds=bands_dir[10]['ds_list'][0])
        x_topleft = x_center - patch_size // 2
        y_topleft = y_center - patch_size // 2

        # round 10m xoff, yoff, patch_size to align with 60m resolution
        x_topleft_align = int(x_topleft // 6) * 6
        y_topleft_align = int(y_topleft // 6) * 6

        # top left in the cropped patch (which is aligned to the 60m resolution)
        y_patch_topleft = y_topleft - y_topleft_align
        x_patch_topleft = x_topleft - x_topleft_align

        target_shape = (patch_size_align, patch_size_align)

        # init array dict for point id
        band_arrays[p_id] = {}
        band_arrays[p_id]['x_topleft'] = x_topleft
        band_arrays[p_id]['y_topleft'] = y_topleft
        for res in bands_dir.keys():
            scale = bands_dir[res]['scale']
            for i in range(len(bands_dir[res]['band_names'])):
                band_name = bands_dir[res]['band_names'][i]
                if read_full_arrays:
                    patch = get_patch(array=bands_dir[res]['band_data_list'][i],
                                      xoff=x_topleft_align // scale,
                                      yoff=y_topleft_align // scale,
                                      win_xsize=patch_size_align // scale,
                                      win_ysize=patch_size_align // scale)
                else:
                    ds = bands_dir[res]['ds_list'][i]
                    patch = read_patch_from_vsis3(xoff=x_topleft_align // scale,
                                                  yoff=y_topleft_align // scale,
                                                  win_xsize=patch_size_align // scale,
                                                  win_ysize=patch_size_align // scale,
                                                  ds=ds)
                if patch.shape != target_shape:
                    if band_name in ['SCL']:
                        order = 0  # nearest
                    else:
                        order = 3  # bicubic

                    patch = resize(patch, target_shape, mode='reflect',
                                   order=order, preserve_range=True).astype(np.uint16)  # bicubic

                # get final patch (with centered gt pixel)
                patch_final = patch[y_patch_topleft:y_patch_topleft + patch_size,
                                    x_patch_topleft:x_patch_topleft + patch_size]

                band_arrays[p_id][band_name] = patch_final

        if maxCount is not None:
            if count == maxCount:
                print('reached maxCount')
                break

    return band_arrays


def read_band(path_band, num_retries=10, max_sleep_sec=5):
    for i in range(num_retries):
        try:
            ds = gdal.Open(path_band)
            band = ds.GetRasterBand(1)
            print('reading full band array...')
            band_array = band.ReadAsArray()
            return band_array
        except: 
            print('Attempt {}/{} failed reading path: {}'.format(i, num_retries, path_band))
            time.sleep(np.random.randint(max_sleep_sec))
            continue
        # raise an error if max retries is reached
    raise RuntimeError("read_band() failed {} times reading path: {}".format(num_retries, path_band))        


def read_sentinel2_bands(data_path, from_aws=False, bucket='sentinel-s2-l2a', channels_last=False):
    bands10m = ['B02', 'B03', 'B04', 'B08']
    bands20m = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'SCL']
    bands60m = ['B01', 'B09']  # 'B10' is missing in 2A, exists only in 1C

    bands_dir = {10: {'band_names': bands10m, 'subdir': 'R10m', 'scale': 1},
                 20: {'band_names': bands20m, 'subdir': 'R20m', 'scale': 2},
                 60: {'band_names': bands60m, 'subdir': 'R60m', 'scale': 6}}

    if '.zip' in data_path:
        archive = ZipFile(data_path, 'r')  # data_path is path to zip file

    band_arrays = {}
    tile_info = None
    for res in bands_dir.keys():
        bands_dir[res]['band_data_list'] = []
        for i in range(len(bands_dir[res]['band_names'])):
            band_name = bands_dir[res]['band_names'][i]

            if from_aws:
                print('Opening bands with gdal vsis3...')
                path_band = os.path.join('/vsis3', bucket, data_path, bands_dir[res]['subdir'], band_name + '.jp2')
            else:
                # get datapath within zip file
                # get path to IMG_DATA
                path_img_data = \
                [name for name in archive.namelist() if name.endswith('{}_{}m.jp2'.format(band_name, res))][0]
                path_band = os.path.join(data_path, path_img_data)
                path_band = '/vsizip/' + path_band

            print('path_band: ', path_band)
            if not tile_info:
                ds = gdal.Open(path_band)
                tile_info = get_tile_info(ds)

            # read all band data to memory once
            band_arrays[band_name] = read_band(path_band=path_band)

    print("Opening CLD band...")
    if from_aws:
        path_band = os.path.join('/vsis3', bucket, data_path, 'qi', 'CLD_20m.jp2')
    else:
        path_img_data = \
        [name for name in archive.namelist() if name.endswith('CLD_20m.jp2') or name.endswith('MSK_CLDPRB_20m.jp2')][0]
        path_band = os.path.join(data_path, path_img_data)
        path_band = '/vsizip/' + path_band
    print('cloud path_band:', path_band)

    band_arrays['CLD'] = read_band(path_band=path_band)

    target_shape = band_arrays['B02'].shape
    print('resizing 20m and 60m bands to 10m resolution...')
    for band_name in band_arrays:
        band_array = band_arrays[band_name]
        if band_array.shape != target_shape:
            if band_name in ['SCL']:
                order = 0  # nearest
            else:
                order = 3  # bicubic

            band_arrays[band_name] = resize(band_array, target_shape, mode='reflect',
                                            order=order, preserve_range=True).astype(np.uint16)
    print('sorting bands...')
    image_array = sort_band_arrays(band_arrays=band_arrays, channels_last=channels_last)
    return image_array, tile_info, band_arrays['SCL'], band_arrays['CLD']


def reproject_raster(src_file_path, out_dir, dst_file_name, ref_file_path, file_format='VRT',
                     resample_algo=gdalconst.GRA_Bilinear, out_nodata=np.nan):
    """
    For max use: gdalconst.GRA_Max
    file_format: 'GTiff', 'VRT', 'MEM'
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    outpath = out_dir + '/' + dst_file_name

    # Open source dataset
    src_ds = gdal.Open(src_file_path)
    print('total raster width, height: ', src_ds.RasterXSize, src_ds.RasterYSize)
    print('number of vhm bands: ', src_ds.RasterCount)

    ref_ds = gdal.Open(ref_file_path)
    print('****reference dataset*****')
    print('projection: ', ref_ds.GetProjection())
    print(ref_ds.GetGeoTransform())
    print('total raster width, height: ', ref_ds.RasterXSize, ref_ds.RasterYSize)

    ref_proj = ref_ds.GetProjection()
    ref_geotransform = ref_ds.GetGeoTransform()
    xRes = ref_geotransform[1]
    yRes = ref_geotransform[5]
    x1 = ref_geotransform[0]
    y1 = ref_geotransform[3]
    x2 = x1 + ref_ds.RasterXSize * xRes
    y2 = y1 + ref_ds.RasterYSize * yRes
    print('x1, x2, y1, y2', x1, x2, y1, y2)
    xmin, xmax = min(x1, x2), max(x1, x2)
    ymin, ymax = min(y1, y2), max(y1, y2)
    print('outputBounds=(xmin, ymin, xmax, ymax)', xmin, ymin, xmax, ymax)

    print('warping...')
    warp_opts = gdal.WarpOptions(format=file_format,
                                     dstSRS=ref_proj,
                                     outputBounds=(xmin, ymin, xmax, ymax),
                                     width=ref_ds.RasterXSize,
                                     height=ref_ds.RasterYSize,
                                     dstNodata=out_nodata)

    gdal.Warp(outpath, src_ds, options=warp_opts, resampleAlg=resample_algo)
    print('{} saved!'.format(outpath))

    warped_ds = gdal.Open(outpath)
    print('*****warped******')
    print('number of vhm bands (warped): ', warped_ds.RasterCount)
    print('projection: ', warped_ds.GetProjection())
    print(warped_ds.GetGeoTransform())
    print('total raster width, height: ', warped_ds.RasterXSize, warped_ds.RasterYSize)

    ref_ds = None
    src_ds = None
    return warped_ds


def translate_vrt_to_tif(input_file):
    """translate vrt to tif with compression"""
    translate_options = gdal.TranslateOptions(format='GTiff', creationOptions=['COMPRESS=DEFLATE', 'PREDICTOR=2',
                                                                               'BIGTIFF=YES', 'NUM_THREADS=8'])
    output_file = input_file.replace('.vrt', '.tif')
    gdal.Translate(output_file, input_file, options=translate_options)


def load_tif_as_array(path, set_nodata_to_nan=True):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)

    array = band.ReadAsArray().astype(float)
    tile_info = get_tile_info(ds)
    # set the nodata values to nan
    nodata_value = band.GetNoDataValue()
    tile_info['nodata_value'] = nodata_value
    if set_nodata_to_nan:
        array[array == nodata_value] = np.nan
    return array, tile_info


def get_reference_band_path(path_zip_file, ref_band_suffix='B02_10m.jp2'):
    archive = ZipFile(path_zip_file, 'r')
    archive_B02 = [name for name in archive.namelist() if name.endswith(ref_band_suffix)][0]
    refDataset_path = os.path.join('/vsizip/' + path_zip_file, archive_B02)
    return refDataset_path


def get_reference_band_ds_gdal(path_file, ref_band_suffix='B02_10m.jp2'):
    if ".zip" in path_file:
        refDataset_path = get_reference_band_path(path_file, ref_band_suffix)
    else:
        # create path on aws s3
        refDataset_path = os.path.join('/vsis3', 'sentinel-s2-l2a', path_file, 'R10m', 'B02.jp2')
    ds = gdal.Open(refDataset_path)
    return ds


def get_ref_file_path(tile_name, image_dir):
    path_zip_file = glob(os.path.join(image_dir, '*{}*.zip'.format(tile_name)))[-1]
    archive = ZipFile(path_zip_file, 'r')
    archive_B02 = [name for name in archive.namelist() if name.endswith('B02_10m.jp2')][0]
    ref_file_path = os.path.join('/vsizip/' + path_zip_file, archive_B02)
    return ref_file_path

