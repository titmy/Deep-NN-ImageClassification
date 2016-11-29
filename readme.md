# Project Title

Deep Neural Network Image Classifier for classifying warehouse products at Lowes to assist in inventory upkeep

## Getting Started

Have tensorflow installed with dependencies. You can use docker container which would speed up development and production. Solves dependencies issues when shipping.

[All Instructions here for docker] (https://www.docker.com)

### Prerequisites

#### Using Docker
[Follow Link] (https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc#0)

What things you need to install the software and how to install them

* Install TF Image
```
docker run -it gcr.io/tensorflow/tensorflow:latest-devel
# cd/tensorflow
gitpull
```
* Link Dataset to TF Image
```
docker run -it -v path_to_files/tf_files/folder_with_images gcr.io/tensorflow/tensorflow:latest-devel
```
* Run pre-written training script in tensorflow/examples
```
# python tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/tf_files/bottlenecks \
--how_many_training_steps 500 \
--model_dir=/tf_files/inception \
--output_graph=/tf_files/retrained_graph.pb \
--output_labels=/tf_files/retrained_labels.txt \
--image_dir /tf_files/flower_photos
```

or

#### Using on machine tensor
[Follow tensorflow website] (https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html#requirements)



### Implementation 

```
python label_image.py path_to_file/filename.jpg
```

## Deployment

Add additional notes about how to deploy this on a live system

