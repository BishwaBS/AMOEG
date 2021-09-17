

# importing some common libraries
import numpy as np
import cv2
import random
import pandas as pd
import argparse
import math
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow


#importing packages related to mask rcnn
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

from detectron2.data import detection_utils as utils
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
import os
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
import copy

#importing packages related to raster handling
import gdal
import os
from osgeo import osr
import rasterio
from rasterio.features import shapes
from rasterio.plot import show
import fiona
import geopandas as gpd
import shapely
from shapely import wkt
from shapely.geometry import Point, Polygon
import glob
from detectron2.data.datasets import register_coco_instances


def define_predictor(modelfile):
    trainname = "train" + str(random.randint(0, 10000000))
    valname = "val" + str(random.randint(0, 100000000))

    register_coco_instances(trainname, {}, os.path.join(os.getcwd(), "detectron2/data/coco/train/train.json" ), os.path.join(os.getcwd(), "detectron2/data/coco/train/images" ))
    register_coco_instances(valname, {}, os.path.join(os.getcwd(), "detectron2/data/coco/val/val.json" ), os.path.join(os.getcwd(), "detectron2/data/coco/val/images"))
    
    
    cfg = get_cfg()

    class CocoTrainer(DefaultTrainer):
      @classmethod
      def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("coco_eval", exist_ok=True)
            output_folder = "coco_eval"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)
        
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (trainname,)
    cfg.DATASETS.TRAIN = (valname,)
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 50000 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (2000, 10000)
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 #your number of classes + 1
    cfg.TEST.EVAL_PERIOD = 1000

    #loading the weights
    cfg.MODEL.WEIGHTS = os.path.join(os.getcwd(),  modelfile)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load()

    #runnin the evaluation
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    return predictor


#reading and converting the raster to acceptable format by CNN models
def read_convert_raster(raster):

    rast=rasterio.open(raster) # opening the raster

    rastarray=rast.read() #reading it so that it is stored in np array format
    
    if rastarray.shape[0]>3: #removing the alpha channel
      rastarray=rastarray[:-1, :,:] 

    b,g,r= rastarray[0,:,:], rastarray[1,:,:], rastarray[2,:,:] # reading individual bands so that np array order can be changed
    bgr=np.stack((b,g, r), axis=2)
    bgr=np.uint8(bgr) # converting to unsigned 8-bit integer so that deep learning model can work on it
    crs=rast.crs
    return bgr, rast, crs

#partitioning the big image into multiple arrays
def partition_raster(bgr, n):
    if bgr.shape[2]!=3:
      print("bgr not loaded correctly")

    w,h=bgr.shape[0], bgr.shape[1] # reading width and height of the raster
    initial_w= 0
    initial_h=0
    n_wh = int(math.sqrt(n))
    step_w= int(w/n_wh)
    step_h= int(h/n_wh)

    imglist=[] #creating an empty list to store partitioned images
    start_coords_list=[] #creating an empty list to store the starting coordinates of these partitioned images for mappint their coordinates to the original images

    for i in range(n_wh):
      
      for j in range(n_wh):
        im= bgr[initial_h: step_h, initial_w:step_w, :]
        start_coords = (initial_w, initial_h)
        initial_w= step_w
        step_w= step_w + step_w

        start_coords_list.append(start_coords)
        imglist.append(im)

      initial_h=step_h
      step_h = step_h + step_h
      initial_w=0
      step_w= int(w/n_wh)
    if len(imglist)==0:
      print("raster not partitioned well")
    print ("...you partitioned your main raster into {} sub rasters...".format(len(imglist)))
    return imglist, start_coords_list

#finding the scalefactor for long and lat by determining how much lat and long change compared to change in pixel numbers
def find_latlong_scale(rast):

    lonScale=(rast.xy(0, 100)[0] -rast.xy(0,0)[0]) /100
    latScale=(rast.xy(0, 0)[1] -rast.xy(100,0)[1]) /100
    return lonScale, latScale


#inferencing over each splitted image or array
def inference_raster(predictor, imglist, rast, start_coords_list, lonScale, latScale):
    idx=0
    df=pd.DataFrame(columns=["long", "lat"]) #creating an empty dataframe for storing all the detected coordinates later
    for id, img in enumerate(imglist): #iterating over partitioned rasters for each big raster scene
      outputs = predictor(img) # applying the predictor instace that was instanstiated above 
      instances = outputs["instances"].to("cpu")
      bboxes=instances.pred_boxes.tensor.cpu().numpy() # convert the tensor formatted bbox info to numpy array for ease later on

      start_x= start_coords_list[id][0] #extracting topleft x and y coordinate of each partitioned raster
      start_y= start_coords_list[id][1]

      #finding the center of bboxes

      if bboxes.shape[0]<1: #moving to next partitioned raster if null detections
        continue
      

      for i in range(bboxes.shape[0]): #iterating over each bboxes detected to find the coordinates of the center point relative to the big raster
        cX, cY = bboxes[i][0]+ int((bboxes[i][2] - bboxes[i][0])/2),  bboxes[i][1]+ int((bboxes[i][3] - bboxes[i][1])/2) #finding the center point of bbox
      
        #mapping the real coordinates of pixels in partitioned raster in respect to whole scene
        real_coord = (start_x + cX, start_y + cY)
        real_gps_coord_long = rast.xy(0,0)[0]+ (real_coord[0] * lonScale)
        real_gps_coord_lat = rast.xy(0,0)[1] - (real_coord[1]* latScale)
        df.loc[idx]= [real_gps_coord_long] + [real_gps_coord_lat]
        idx+=1
    return df


def dftoshapefile (df, outputdir, crsdata, plotname):
      

    outputname=os.path.join(os.getcwd(), outputdir, plotname +".shp" )

    ##convert the longitude and latitude to shapely point format
    geometry = [Point(xy) for xy in zip(df["long"], df["lat"])]

    ##converting them to geopandas df
    gdf = gpd.GeoDataFrame(df,geometry=geometry, crs=crsdata)
    gdf.to_file(outputname, driver='ESRI Shapefile') #expoerting geodataframe to shapefiles
    return gdf

#Calling all the functions to get the work done

def apply_model_mosaic(raster_dir, partition_number, modelfile, outputdir):
    predictor=define_predictor(modelfile)
    print("....................finished installing and loading packages...............................")
    print("....................starting inferencing process...............................")

    
    for raster in glob.glob(os.path.join(os.getcwd(), raster_dir) +  "/*.tif"): #iterating over all the tif files found in the raster directory       

          plotname=os.path.split(raster)[1].rsplit('.', 1)[0] # extracting a name for raster files for storing it later to the df

          print("Loading {}".format(str(plotname))) 
          bgr, rast, crsdata=read_convert_raster(raster)
          print(rast)
          imglist, start_coords_list = partition_raster(bgr, partition_number) 
          lonScale, latScale = find_latlong_scale(rast)
          print("starting inferencing process for {}".format(plotname))
          df = inference_raster(predictor, imglist, rast, start_coords_list, lonScale, latScale)
          gdf=dftoshapefile(df, outputdir, crsdata, plotname)
          print(gdf)
          print("shapefile has been exported successfully for {}". format(plotname))
          # if visualize_mode==True:
          #     print("mode enabled")
          #     fig, ax = plt.subplots(figsize=(15, 15))
          #     show(rast.read([3,2,1]), transform =rast.transform, ax=ax)
          #     gdf.plot(ax=ax, facecolor='none', edgecolor='red', aspect = 1)
          #     plt.show()
          # break

    print("..............process completed..........................")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')    


def main():
    parser = argparse.ArgumentParser(description="Applies the CNN model over Orthomosaick and obtain coordinates")
    parser.add_argument("raster_dir", help="Relative location of the directory where mosaicked imagery are", type=str)
    parser.add_argument("partition_number", help="Number of partitioned rasters to be made out of raster", type=int)
    parser.add_argument("modelfile", help="Relative location of trained model file", type=str)
    parser.add_argument("outputdir", help="Relative location of the output directory where shapefiles are to be stored", type=str)
    # parser.add_argument("visualize_mode", help="Set to True if you want to see coordinates being plotted each respective raster", default="False", action="store_true")
    args, unknown = parser.parse_known_args()
    apply_model_mosaic(args.raster_dir, args.partition_number, args.modelfile, args.outputdir)      

if __name__=="__main__":
    main()
