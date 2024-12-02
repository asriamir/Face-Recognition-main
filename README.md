# Face Recognition with Deep Learning

This repository contains an implementation of **Face Verification** and **Face Recognition** using a pre-trained **FaceNet** model and TensorFlow.

## Overview

This project demonstrates face verification and recognition using deep learning. A pre-trained **FaceNet** model is used to extract face embeddings, and these embeddings are then used for comparing faces for verification or recognition. It includes:
- **Face Verification**: Verifies if the person in a given image matches a known identity.
- **Face Recognition**: Identifies the person from a database of known faces.

The model uses **triplet loss** for face verification and **nearest neighbor** search for face recognition.

## Requirements

Before running the code, make sure to install the necessary dependencies:

- Python 3.6 or higher
- TensorFlow 2.x
- Keras
- NumPy
- OpenCV
- Pillow (PIL)
- Matplotlib

You can install these dependencies with:

```bash
pip install tensorflow numpy opencv-python Pillow matplotlib
```

### Setup and Data Preparation
Clone the Repository

To clone this repository, run:

```bash
git clone https://github.com/your-username/Face-Recognition.git
cd Face-Recognition
```


### Prepare the Image Dataset

Collect images of individuals to verify or recognize. Save these images in an images/ directory. Some example images of individuals are included in the repository, but you can replace them with your own dataset.

### Load the Pre-trained Model

This project uses a pre-trained FaceNet model. You can download the model weights and JSON from an external source or train your own FaceNet model. The model is loaded using Keras and consists of:

model.json: The model architecture.
model.h5: The model weights.


## Usage
### 1. Face Verification
To verify a person in an image against a known identity, use the verify() function. This function takes an image path and an identity name to compare. It returns whether the person in the image matches the given identity from the database.

Example usage:
```bash
verify("images/camera_2.jpg", "kian", database, FRmodel)
```

The output will indicate whether the person in the image is the same as the provided identit

```bash
It's kian, welcome in!
```

or:

```bash
It's not kian, please go away

```


### 2. Face Recognition
To recognize a person from a given image, use the who_is_it() function. This function compares the face in the image to the database of known faces and returns the identity with the smallest distance.

Example usage:

```bash
who_is_it("images/camera_0.jpg", database, FRmodel)
```

### 3. Loss Function
The project uses the triplet loss function, which is designed to ensure that the model minimizes the distance between positive pairs (same identity) and maximizes the distance between negative pairs (different identities).

Here is the code for the loss function:

```bash
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.maximum(tf.add(tf.subtract(pos_dist, neg_dist), alpha), 0)
    loss = tf.reduce_sum(basic_loss)
    return loss
```

### 4. Database of Known Faces
The face embeddings of known individuals are stored in the database dictionary. Each key is the person's name, and the corresponding value is the face embedding (a vector representation).

Example:

```bash
database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
```

### 5. Model Summary
You can print the summary of the loaded FaceNet model by uncommenting the following line in the code:

```bash
# model.summary()
```

## Example of Verifying a Person

```bash
verify("images/camera_2.jpg", "kian", database, FRmodel)
```

### output:

```bash
It's kian, welcome in!
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.










