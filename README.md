# TensorFlow Models

This repository contains a number of different models implemented in [TensorFlow](https://www.tensorflow.org):

The [official models](official) are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.

The [research models](https://github.com/tensorflow/models/tree/master/research) are a large collection of models implemented in TensorFlow by researchers. They are not officially supported or available in release branches; it is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.

The [samples folder](samples) contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.

The [tutorials folder](tutorials) is a collection of models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).

## Contribution guidelines

If you want to contribute to models, be sure to review the [contribution guidelines](CONTRIBUTING.md).

## License

[Apache License 2.0](LICENSE)

## Object detection using own Dataset
Clone this into your local machine:
'''
git clone https://github.com/mohanrobotics/models.git
'''

### Instructions:
###### Step 1(getting the images in the right place): 
In the '/models/research/object_detection' directory create a folder 'images' and two sub-folder'train' and 'test' inside 'images' folder. Keep the image name in numbers(e.g 1.jpg , 2.jpg.. )
images
	- train (consist of train images)
	- test (consist of test images)
	- (all the images from train and test put together here)
###### Step2:
 LabelImg is used to annotate the image. LabelImg can be installed using 'https://github.com/tzutalin/labelImg'.  Launch LabelImg and open the directory where we have placed the train and test images. First open the dir with train images. Click 'Create RectBox' and annotate the image and label them. Click 'save' which will save an .xml file for each image. Annotate the images in the 'test' folder as well.
###### Step3 :
 In the '/models/research/object_detection' directory, Launch the xml_to_csv.py which will generate the csv file for the train and test image in the 'data' folder.
###### Step 4: 
Install the Tensorflow Object Detection API following the step given in the 'https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md'

Hint : apart from dependencies,# From tensorflow/models/research/, the following line are important.(when you get an error:No module named 'object_detection')
	- wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
	- unzip protobuf.zip
	- ./bin/protoc object_detection/protos/*.proto --python_out=.
	- protoc object_detection/protos/*.proto --python_out=.
	- export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

###### Step5 : 
After the successful installation, following changes neeed to be done in the 'generate_tfrecord.py' from  the '/models/research/object_detection' directory. 
	In the generate_tfrecord.py, change the 'row_label' with the name of the label used while annotating in the 'class_text_to_int' function.
	Now launch the generate_tfrecord.py file as 'python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record' for training set and 'python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record' for the test set

###### Step 6 : 
Edit the 'object_detection.pbtxt' from the '/models/research/object_detection/data' with your label and id. In my case, id is 1 and label is 'mohan'
###### Step 7 :
 'ssd_mobilenet_v1_pets.config' has all the configuration required for training. You can adjust the num_classes, batch_size , number of training steps etc., in this file.(I have set the batch size to 12 and num_steps: 200000. I am having nvidia gforce 940mx). So based on you GPU capacity , you can vary this.
###### Step 8 : 
From the '/models/research/object_detection/legacy' , run 'python3 legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config'. Now training will run. This will take time depending upon the gpu capacity, batch size and num_steps. In my case with nvidia 940mx gpu , batch size - 12 and num_steps - 20000 took nearly 5 hours. Loss should go less than 1.
###### Step 9 :
 Now we need to create the inference graph. Run 'python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix training/model.ckpt-10856 --output_directory inference_graph'  from the '/models/research/object_detection' directory. Edit the model.ckpt number with ckpt generated under '/models/research/object_detection/training'. Give the ckpt number which is high.

###### Step 10:
 Now you can put some some images in the models/research/object_detection/test_images folder that you want to test with your trained model. After placing the images, launch 'jupyter notebook' and open the 'object_detection_tutorial.ipynb' file. In the detection section,give the range of your filenames you will be tested. In the following , images 3.jpg,4.jpg,5.jpg,6.jpg,7.jpg under the test_images folder will be tested.
'''
	TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.jpg'.format(i)) for i in range(3, 8) ] 
'''

