#-------------------------------------------------------------------------------
# Name:        Beaker Volume Detection
# Purpose:     To determine the volume of 2 liquids in a beaker
#
# Python ver:  2.7.12
# Author:      Josh
#
# Created:     20/04/2017
# Copyright:   (c) Josh 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import sys
import numpy as np
import cv2 # Use version 3.2.0 of OpenCV

def ocr(testImage, trainImageFilename, numTrainImages):
    # Performs Optical Character Recognition (OCR) by taking the input binary (black and white) testImage and
    # comparing it to the trainImageFilename which contains the image(s) of interest. This comparision is done
    # using the k-nearest neighbours method and the trainImage which has the shortnest distance to the testImage
    # is the one that it is likely to be. Then the array train_labels gives the trainImages a digital value (ie. 0-9 or a-z)
    # so that the computer knows what that character is. This value and the distance are then output by this function

    # Import the training image from the trainImageFilename provided
    trainImg = cv2.imread(trainImageFilename, 0)

    # Splits the image to numTrainImages number of cells, each 20x20 size
    cells = [np.hsplit(row,1) for row in np.vsplit(trainImg,numTrainImages)]

    # Turn it into a Numpy array, size = (numTrainImages,1,20,20)
    x = np.array(cells)

    # Prepare train and newcomer (test) images by reshaping the 20x20 image into a more compact 1x400 image
    train = x[:,:100].reshape(-1,400).astype(np.float32) # Size = (10, 400)
    newcomer = testImage.reshape(-1,400).astype(np.float32) # Size = (1, 400)

    # Create labels for train images to be output
    train_labels = np.arange(numTrainImages)

    # Initiate kNN, train the data, then test it with test data for k=1, meaning it only considers the nearest neighbour
    knn = cv2.ml.KNearest_create()
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    ret,result,neighbours,dist = knn.findNearest(newcomer,k=1)
    retdist = [int(ret), int(dist[0,0])]

    # Return a two value vector (predicted number, confidence of prediction)
    return (retdist)

def boundingBoxes(image, hRange, wRange, minContourArea, imageCopy):
    # Takes in a grayscale image to find the contours that outline the edges between colours in the image using OpenCV's findContours
    # function. hRange and wRange are 2 element vectors which define the [min, max] of the height and width of features of interest.
    # minContourArea defines the minimum area of a contour of a feature of interest to remove noise. This then outputs an array
    # containing the x and y position and h and w of every contour found in the image
    # The copy image is to draw the bounding boxes on without affecting later detection to show the results visually
    _, contours, _ = cv2.findContours(image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    retVal = np.array([], dtype=int).reshape(0,4)

    # Iterate over all the contours found to remove the ones that are not of interest
    for cnt in contours:

        # First remove all contours that have an area that is too small using the built-in function contourArea
        if cv2.contourArea(cnt)>minContourArea:

            # Define variables for the remaining positions and dimensions of the bounding box surrounding the contour
            [x,y,w,h] = cv2.boundingRect(cnt)

            # Concatenate the position and dimnesion of the bounding box to the array to be returned by the function
            # to create a list containing all of the bounding boxes of interest in the image and draw them on the
            # copy image to visually show the areas of interest
            if  h>hRange[0] and h<hRange[1] and w>wRange[0] and w<wRange[1]:

                retVal = np.vstack([retVal, map(int,[x,y,w,h])])
                cv2.rectangle(imageCopy,(x,y),(x+w,y+h),(0,0,255),2)

    return (retVal)

# Import the image of the beaker containing the two fluids and find it's dimensions
im = cv2.imread('compVisTest2.jpg')
imCopy = im.copy()
dimIm = im.shape

#################      Finding Volume Number Marks (mL)      ###################

# First grayscale the image to allow for easier image processing, then blur it to reduce noise and widen the edges of
# the numbers to make them easier to detect, then threshold the image to turn the grey image into a strictly black and
# white image using the build-in adaptive threshold which is optimized for images that have different lighting conditions
# to select the dark text across the image
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
numBlur = cv2.GaussianBlur(gray,(7,7),0)
numThresh = cv2.adaptiveThreshold(numBlur,255,1,1,11,2)

# Gather the positions of all possible features on the image that could be a volume digit, these are about square and are at
# most 30px so this fact helps eliminate some of the noise that are could not be pixels
# Create the newMark array which will be 2 column wide and temporarily contain the volume marks detected and concatenate them to the
# volumeMarks array which will contain all of these marks
boundingBoxPos = boundingBoxes(numThresh, [5,30], [3,30], 50, imCopy)
newMark = np.array([], dtype=int).reshape(0,1)
boundingBoxCounter = 0

while boundingBoxCounter < len(boundingBoxPos):

    x, y, w, h = boundingBoxPos[boundingBoxCounter, 0:4]

    # Cut out the region of interest from the picture plus a 2 pixel border all around to improve OCR accuracy
    roi = numThresh[y-2:y+h+2,x-2:x+w+2]

    # Reshpae this region of interest to 20x20 pixels for improved OCR processing reliability
    roismall = cv2.resize(roi,(20,20))

    # Pass the region of interest into the ocr function to return the predicted value and the confidence
    retdist = ocr(roismall, "0-9.png", 10)

    # If the confidence (lower is better) is below 4000000 then it is likely to be the predicted digit
    if retdist[1] < 4000000:

        newMark = np.vstack([newMark, retdist[0]])
        boundingBoxCounter += 1

    else:
        # Clear any row that is likely not a digit and repeat that same count of boundingBoxCounter to do the next row
        boundingBoxPos = np.delete(boundingBoxPos, boundingBoxCounter, 0)

volumeMarks = np.hstack([boundingBoxPos, newMark])

#########   Grouping adjacent digits together to form full numbers   ###########

# Copy the array containing the position of the digits to manipulate it without losing the info for the location
# of the digits. Then initialize the array that will contain the volume numbers and their position
tempAry = np.copy(volumeMarks)
volumeMarksValue = np.array([], dtype=int).reshape(0,3)

# Iterate through all of the digits to determine which other values in the list of digits are adjacent both
# horizontally and vertically as this means they should be combined as one larger number
for i in xrange(len(tempAry)):

    newValue = np.array([int(tempAry[i,4])])

    # Check if row is empty which indicates that it has been cleared previously and can be skipped
    if sum(tempAry[i,:]) == 0:
        continue

    else:

        for j in xrange(i+1,len(tempAry)):

            # Condition for when there is nothing around the digit then break out of the loop
            if tempAry[i,1] - tempAry[i,3] > tempAry[j,1] or tempAry[i,1] + tempAry[i,3] < tempAry[j,1] or tempAry[i,0] - 4 * tempAry[i,2] > tempAry[j,0]:
                break

            # Check for if row has already been cleared and can be skipped
            elif sum(tempAry[j,:]) == 0:
                continue

            # Condition for when there is a digit close by
            else:

                # Determining if the digit is to the left or right of the first digit then concatenating the digit
                # to the left or right accordingly and then clear that row so it is not checked again

                if tempAry[i,0] > tempAry[j,0]:
                    newValue = np.hstack([int(tempAry[j,4]), newValue])
                    tempAry[j,:] = 0
                else:
                    newValue = np.hstack([newValue, int(tempAry[j,4])])
                    tempAry[j,:] = 0

    # If it is only a 1 or 2 digit number add zeros to the front so it becomes 3 digits and can be concatenated without changing its value
    if len(newValue) < 3:
        newValue = np.hstack([0, newValue])
        if len(newValue) < 3:
            newValue = np.hstack([0, newValue])

    # Concatenate the new number to the bottom of the array containing the values of the volume markings
    volumeMarksValue = np.vstack([volumeMarksValue, newValue])

# Remove all rows that contain zeros in the array so that the positions of the volume markings can be concatenated to the array
# containing their values
tempAry = tempAry[~np.all(tempAry == 0, axis=1)]

# Convert all integers into strings using map and then join the strings in each row together to form one number
for i in xrange(len(volumeMarksValue)):

    volumeMarksValue[i] = (''.join(map(str, volumeMarksValue[i])))
    volumeMarksValue[i,1] = tempAry[i,1]

print("The Detected Volumes Markings are:" , volumeMarksValue[:,0])

######################      Finding Volume Ticks      ##########################

# Blur the image less to still remove noise but keep the edge between the phases clear and convert to HSV colour space
# as this is easier to create a range of similar colours since the hue stays almost the same, only the saturation
# and value changed across an area of colour
blur = cv2.GaussianBlur(im,(3,3),0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

# Define the range of colours of the ticks in HSV colour space, ADJUST THESE VALUES TO DETECT TICKS THAT ARE DIFFERENT COLOURS
lower_blueTick = np.array([70,20,20])
upper_blueTick = np.array([130,255,255])

# Threshold the HSV image to get an area in white that is where there is the colour of the ticks in the original image
thresh = cv2.inRange(hsv, lower_blueTick, upper_blueTick)

# Gather the positions of all possible features on the image that could be a volume tick, these are long and thin
# so this fact helps eliminate a lot of the noise by setting lengh to 35-50px and height to only 4-10px
ticksBoundingBoxPos = boundingBoxes(thresh, [4,10], [35,50], 50, imCopy)
newTickMark = np.array([], dtype=int).reshape(0,1)
boundingBoxCounter = 0

while boundingBoxCounter < len(ticksBoundingBoxPos):

    x, y, w, h = ticksBoundingBoxPos[boundingBoxCounter, 0:4]

    # Cut out the region of interest from the picture plus a 2 pixel border all around to improve OCR
    roi = thresh[y-2:y+h+2,x-2:x+w+2]

    # Reshpae this region of interest to 20x20 pixels for OCR processing
    roismall = cv2.resize(roi,(20,20))

    # Pass the region of interest into the ocr function to return the predicted value and the confidence
    retdist = ocr(roismall, "tick.png", 1)

    # If the confidence (lower is better) is below 5000000 then it is likely to be a tick
    if retdist[1] < 5000000:

        boundingBoxCounter += 1

    else:
        # Clear any row that is likely to not be a tick and repeat the count
        ticksBoundingBoxPos = np.delete(ticksBoundingBoxPos, boundingBoxCounter, 0)

# The average height of letters used for the marking of the numbers which will be used to determine if a tick is adjacent to a number
averageLetterHeight = np.average(volumeMarks[:,3])
tickCounter = 0
while tickCounter < len(ticksBoundingBoxPos):

    if len(ticksBoundingBoxPos) != len(volumeMarksValue):
        print('The number of volume markings detected is not equal to the number of tick marks, the volume calculated may not be accurate, please adjust the lower_blue and upper_blue value or the numBlur to another odd number besides 7,7')

    # If the tick is within half the average height of a number from one of the volume markings then assign it to that number, if not
    # then it is probably a false poisitve and that row should be deleted and that row repeated
    if ticksBoundingBoxPos[tickCounter,1] + 0.5 * averageLetterHeight > volumeMarksValue[tickCounter,1]:
        if ticksBoundingBoxPos[tickCounter,1] - 1.5 * averageLetterHeight < volumeMarksValue[tickCounter,1]:
            volumeMarksValue[tickCounter,2] = ticksBoundingBoxPos[tickCounter,1] + 0.5 * ticksBoundingBoxPos[tickCounter,3]
            tickCounter += 1
    else:
        ticksBoundingBoxPos = np.delete(ticksBoundingBoxPos, tickCounter, 0)

####################      Finding Height of Liquids      #######################

# Find the average spacing between ticks and the volume increment that corresponds to for this beaker to use to find how
# many ticks the height of each liquid is and the volume that corresponds to
posOfTicks = volumeMarksValue[:,2]
averageTickSpacing = abs(np.mean([x - posOfTicks[i - 1] for i, x in enumerate(posOfTicks)][1:]))
volumePerTick = volumeMarksValue[1,0] - volumeMarksValue[0,0]

# Find the bounding box for each liquid layer which should only return one box as these are quite large
blur = cv2.GaussianBlur(im,(15,15),0)

# Convert BGR to HSV
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# Define range in HSV for each liquid phase, ADJUST THESE VALUES TO DETECT LIQUIDS OF DIFFERENT COLOURS
lower_water = np.array([63,50,50])
upper_water = np.array([67,255,255])
lower_oil = np.array([20,50,50])
upper_oil = np.array([30,255,255])

# Threshold the HSV image to produce an image where only liquid phases are shown in white
mask_water = cv2.inRange(hsv, lower_water, upper_water)
mask_oil = cv2.inRange(hsv, lower_oil, upper_oil)

# Locate the bounding boxes for the liquids by starting with a box that is just big enough to fit in image and then stepping down
# the minimum width in steps of 10 pixels until it is just below the thickness the that layer of liquid
waterBoundingBoxSize = [min(dimIm[0:1])-10,min(dimIm[0:1])]
oilBoundingBoxSize = [min(dimIm[0:1])-10,min(dimIm[0:1])]
while True:
    waterBoundingBox = boundingBoxes(mask_water, waterBoundingBoxSize, [250, dimIm[1]], 50, imCopy)
    oilBoundingBox = boundingBoxes(mask_oil, oilBoundingBoxSize, [250, dimIm[1]], 50, imCopy)
    if waterBoundingBox.size == 0:
        waterBoundingBoxSize[0] -= 10

    if oilBoundingBox.size == 0:
        oilBoundingBoxSize[0] -= 10

    if waterBoundingBox.size > 0 and oilBoundingBox.size > 0:
        break

    if waterBoundingBoxSize[0] <= 0 or oilBoundingBoxSize[0] <= 0:
        print('No liquid layer detected, please adjust the lower_oil, lower_water, upper_water or upper_oil value to match the colour of the liquids')
        sys.exit()

# Find the median height of the water pixels by summing the pixels in every column and then dividing by 255 as this
# is the value stored for each white pixel, which is what represents where that phase is in each picture. Then take
# the maximum length vertical line to find the height of that phase since some lines will be too short due to noise
# and the volume markings. Then divide this by tick spacing and multiply by volume per tick to find the volume of each phase
x, y, w, h = waterBoundingBox[0, 0:4]
roiWater = mask_water[y:y+h,x+20:x+w-20]
heightOfWaterPixels = roiWater.sum(axis=0)
waterHeight = np.max(heightOfWaterPixels) / 255
numTicksWater = waterHeight / averageTickSpacing
volumeWater = numTicksWater * volumePerTick
print('Volume of water layer is: %dmL' %volumeWater)

x, y, w, h = oilBoundingBox[0, 0:4]
roiOil = mask_oil[y:y+h,x+20:x+w-20]
heightOfOilPixels = roiOil.sum(axis=0)
oilHeight = np.max(heightOfOilPixels) / 255
numTicksOil = oilHeight / averageTickSpacing
volumeOil = numTicksOil * volumePerTick
print('Volume of oil layer is: %dmL' %volumeOil)

# Save the determined volumes to a text file for potential use by another program
np.savetxt('WaterOilVolume.txt', (volumeWater, volumeOil), fmt='%i')

# Show the provided image with the areas of interest found by this script drawn overtop in red
cv2.imshow('norm', imCopy)
cv2.waitKey()
cv2.destroyAllWindows()
