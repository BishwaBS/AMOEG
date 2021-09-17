This repo runs the MaskRCNN model over the orthomosaick imageries to detect of target of interest, extracts the gps coordinates for the detected objects from the imagery, and export them to shapefiles.
Lot of the tutorial and github resources pertain to inference over single non geotiff image. In contrary, this repo applies the already trained model to orthomosaick imagery in geoTIFF format. User can run 
batch of orthomosaick imagery in a single run; it's just that user has to put all the orthomosaick imageries into a same folder.

**Limitations:**
Only RGB (three bands) ortho mosaick is supported at this time
This repo has not been tested over imagery that has black pixels outside of boundaries (These black pixels appear when the imagery are not in rectanglur shape)

**How to use:**

**If you are using colab:**

**step1:** Mount your google drive 
      ```from google.colab import drive
      drive.mount('/content/drive')```
      
**step2:** clone the github repository

**step3:** set your cloned directory as current working directory
      ```import os```
      ```os.chdir("path to your cloned directory")```

**step4:** install the packages using following command
      ```!pip install -r requirements.txt```

**step5:** run the process
      ```!python apply_maskrcnn_orthomosaic.py --raster_dir <path to your raster directory> --modeldir <relative path to your modeldirectory that contains .pth and metrics.json files --partition number <int> (specify value (int) based on how many splits you want to make the mosaick imagery into" --outputdir <relative path to output directory where you want to save your shapefiles```
      
**If you are using your local machine**

First create a virtual environment using conda or other parties. After virtual environmet has been setup, follow all the steps from **step 2** as mentioned above
