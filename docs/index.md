# CBottle

## Authors
- Jasper Ketelaar, j.ketelaar@student.tudelft.nl, 4453050
- Matthijs van Wijngaarden, ..., ...

## Introduction
CBottle is a project for the seminar course 'Computer Vision by Deep Learning'. The task for the course is to devise our own
computer vision project. The project is inspired by ![Vivino](http://vivino.com/), a wine webshop that has a very nice mobile
application which is able to identify wine based on its label. The exact technology used by vivino is not publically available
and we aim to take our own approach based on the techniques we learn during the seminar course.

The end goal is identify bottles in an image and map them to a certain SKU based on the wine webshop. 
[Grandcruwijnen](https://www.grandcruwijnen.nl). Then a user is able to take a picture of one or more bottle(s) 
and have the pipeline identify the wines and be able to link you to the website's url for that specific wine.

We are primarily going to focus on the computer vision/deep learning aspects but will try to have a working base 
version of the complete pipeline as well.

## Dataset
We have a dataset available to us that consists of high quality pictures of bottle from the Grandcruwijnen website. These
pictures are taken from multiple angles in optimal lighting conditions. The dataset is a lot more sparse than Vivino who
are in possession of a massive community taking a lot of pictures of wine labels and can therefore take a different approach 
to bottle identification. 

## Methodology
The final methodology we took to achieve the results of this project is a pipeline of approaches. We will describe the approaches in terms of a pipeline
both when it comes to the challenges for learning and the challenges for implementation. The first challenge we faced after
discussion with our TA is that the dataset we want to use does not have that many images for each wine (12 construction a 360 degree presentation) and is taken
in perfect lighting conditions. We therefore discussed how to go about implementing this and we landed on a transfer learning approach using data augmentation.

### Augmentation
We need to use our perfect scenario bottle images to augment into more contextual situations that a user would actually take a bottle in. To do this
we wrote an augmentation implementation using MaskRCNN \[ref] and the OpenCV library for Python. The augmentation takes a pre-trained version of MaskRCNN
on the ImageNet dataset and uses that to detect all different sorts and types of bottles in our handpicked augmentation base set from the COCO dataset.

The augmentation base images are then used to augment a random selection of bottles into such that there are now wine bottles where the other sorts of bottles
used to be. The idea behind this augmentation to create a training set is that we can use it to create a custom MaskRCNN model that is capable of detecting different
sorts of wine bottles and find their mask. 

We created an interface to manually select the augmentation results that are good and seem contextually possible to prevent the MaskRCNN model picking up on artifacts
of bad augmentation using our method such as a lot of black pixels due to a bottle not quite fitting in the cut out mask. After discussing with our TA we found out
that we need to take care not to create any other context associations through augmentation and have to take care this is not present in our dataset for training. An example
given to us of such a context association is there being a base image of a fridge that only fits the augmentation of a more thick bottle; such an augmentation could then cause
a model to perhaps learn that a fridge means there must be a thick wine. We therefore make sure to try and select a subset of wines with variable shapes and create enough different augmentations of the same base context.

An example of a base image and an augmented image:
![base](img)

An example of a base image with bad augmentation:
![base](img)

### The MaskRCNN model
We make use of MaskRCNN as the basis for our wine bottle detection. We researched some different implementations that train MaskRCNN as a binary shape classification for one specific class as the basis for our implementation. An example of such an implementation is \[ref]. We cannot directly use the same methodology as the pretrained network we want
to base our model on already has a class for the bottle in general which also encapsulates the class of a wine bottle. An interesting research question that we aim to answer
in our results, after discussing with our TA, becomes: "Can a pretrained network passively forget how it detects a class?". This arises because we want to train the model to
recognize only wine bottles whereas the pretrained weightgs will recognize wine bottles as belonging to the class 'bottle' as well as many other forms and shapes of bottles.

The approach we take with our augmented dataset is to also find images that contain different types of bottles that the pre-trained network recognizes as the bottle type. These
images are then augmented to contain a) a wine bottle and b) a different bottle type such that the network will be forced to train to recognize the wine bottle as a wine bottle
and not recognize the other bottle type as a wine bottle. This is the passive approach we describe in the previous paragraph. An active approach would be to take the pretrained model and create two classes instead of 1 class (excluding the background class), these two classes would then be "wine bottle" and "other bottle" such that you could actively
train the network to learn to recognize them separately. We wanted to take the passive approach as the active approach has already seen plenty of research and this passive approach could find some interesting results that would be of value.

The MaskRCNN model is used to cut out the wine bottle masks which are fed to the next step of our complete application pipeline. We discussed with our TA why we wanted to
use keypoint detectors as a solution to the initially posed problem rather than a classification network. The reason that we agreed on is that the problem requires a model
that can easily be modified to include/exclude wines that were previously a part of the classification process, since there is a frequent rate of this happening. The company
has an almost daily change in this, new wines get added frequently and old wines get disabled frequently.

### Keypoint Detectors
The keypoint detectors idea is an effective way to use the cropped out bottle masks. The main keypoint detectors we researched for our approach were SIFT, SURF and ORB. Since SIFT is scale invariant and implemented readily available in the OpenCV library this was the model we ended up going with. We need to have a model that is flexible to changes and keypoint descriptor vectors can be stored and indexed rather efficiently. 

We use the base images from our wine bottle dataset to create an index tree in the FAISS \[ref] library. We created a process that can produce such an index by downloading all
the images for the active store, computing the descriptor vectors using the SIFT method and indexing these to a specific ID that can be mapped back to the product SKU of the store.

Then for the pipeline we forward the MaskRCNN wine bottle mask with the highest score (for when multiple wine bottles are detected) to the keypoint detection stage. At this stage we perform keypoint detection only on the image pixels that fall within the mask such that no background information is used for the matching of the bottles. 
The keypoint descriptors are forwarded to the indexed model where a nearest-neighbour search is performed. Since the descriptors are separate the nearest-neighbour results are aggregated as a list of ids, counts where the ids can be mapped to a wine sku and the counts are how often that id is encountered as the nearest neighbour.
The highest count nearest neighbour is then returned as the result by the model and this is the result that can then be used to be sent back as a server response in order to complete the implementation of the complete problem described in the introduction.

## Results


## Conclusion
