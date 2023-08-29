"""
from https://github.com/langnico/gdal_processing
"""
from osgeo import gdal, gdalconst
import os
import sys

from src.utils.gdal_process import translate_vrt_to_tif, reproject_raster, get_ref_file_path, RESAMPLE_ALGO_DICT


if __name__ == "__main__":
    print(str(sys.argv))
    
    tile_name = sys.argv[1]
    image_dir = sys.argv[2]
    input_file_path = sys.argv[3]
    out_dir = sys.argv[4]
    out_nodata = sys.argv[5]
    resample_name = sys.argv[6]

    resample_algo = RESAMPLE_ALGO_DICT[resample_name]

    # ********************

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('********************************')
    print('tile_name: ', tile_name)
    print('input_file_path: ', input_file_path)
    dst_file_name = os.path.basename(input_file_path).strip('.tif').strip('.vrt')
    dst_file_name = dst_file_name + '_reprojected_10m_{}_{}.vrt'.format(resample_name, tile_name)
    print('dst_file_name:', dst_file_name)

    print('getting 10m reference band from sentinel2 image_dir...')
    ref_file_path = get_ref_file_path(tile_name=tile_name, image_dir=image_dir)
    print('ref_file_path: ', ref_file_path)

    print('warping to vrt...')
    reproject_raster(input_file_path, out_dir, dst_file_name, ref_file_path, resample_algo=resample_algo)

    print('translating vrt to tif (with compression) ...')
    input_file = os.path.join(out_dir, dst_file_name)
    translate_vrt_to_tif(input_file)

