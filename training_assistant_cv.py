"""
A module that creates yolo model.

"""
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import keras
from typing import Tuple
import matplotlib.pyplot as plt
from anchors_generator import AnchorsGenerator
from helpers import iou_distances


class YoloLoss(tf.keras.losses.Loss):
    def __init__(self, name="yolo_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """ Custom loss function.

        Args:
            y_true (tf.Tensor): true value.
            y_pred (tf.Tensor): predicted value.

        Returns:
            tf.Tensor: combination of localization_loss, obj_loss and class_loss.
        """
        true_box = y_true[:, 0:4]
        pred_box = y_pred[:, 0:4]
        
        true_obj = y_true[:, 4:5]
        pred_obj = tf.sigmoid(y_pred[:, 4:5])
        
        true_class = y_true[:, 5:]
        pred_class = tf.nn.softmax(y_pred[:, 5:], axis=-1)

        obj_mask = tf.cast(true_obj > 0, tf.float32)

        xy_loss = obj_mask * tf.square(true_box[:, 0:2] - tf.sigmoid(pred_box[:, 0:2]))
        wh_loss = obj_mask * tf.square(true_box[:, 2:4] - pred_box[:, 2:4])
        localization_loss = tf.reduce_sum(xy_loss + wh_loss)

        obj_loss = tf.keras.losses.binary_crossentropy(true_obj, pred_obj)
        obj_loss = tf.reduce_sum(obj_loss)

        class_loss_raw = tf.keras.losses.categorical_crossentropy(true_class, pred_class)
        class_loss = obj_mask * tf.expand_dims(class_loss_raw, axis=-1)
        class_loss = tf.reduce_sum(class_loss)

        total_loss = (5.0 * localization_loss) + (1.0 * obj_loss) + (1.0 * class_loss)
        
        return total_loss


class TrainingAssistant():

    def __init__(self, anchors: np.ndarray, input_shape: Tuple=(360, 640, 3), grid_size: Tuple=(9, 16), stride: int=40, num_classes: int=17) -> None:
        
        self.project_dir: str = os.path.dirname(os.path.abspath(__file__))
        self.training_set_dir: str = os.path.join(self.project_dir, "Training set")
        self.models_dir: str = os.path.join(self.project_dir, "Models")

        self.input_shape: Tuple = input_shape
        self.anchors: np.ndarray = anchors
        self.grid_size: Tuple = grid_size
        self.stride: int = stride
        self.num_classes: int = num_classes

    
    def get_grid_count(self, grid_coords: np.ndarray, anchors_count: int) -> np.ndarray:
        """ Computes the occurrence index of each grid cell coordinate within the input
            array. Coordinates that appear multiple times receive consecutive counts
            starting from 1. The count is capped at anchors_count.

        Args:
            grid_coords (np.ndarray): Array of shape (N, 2) containing grid cell coordinates in the form [x, y].
            anchors_count (int):  Maximum count value.

        Returns:
            np.ndarray: Array of shape (N,) containing the occurrence index for each coordinate in the original order.
        """
        sort_idx: np.ndarray = np.lexsort((grid_coords[:, 0], grid_coords[:, 1]))
        coords_s: np.ndarray = grid_coords[sort_idx]

        is_new: np.typing.ArrayLike = np.concatenate(([True], np.any(coords_s[1:] != coords_s[:-1], axis=1)))

        total_count: np.typing.ArrayLike = np.arange(len(coords_s))
        group_offsets: np.typing.ArrayLike = np.maximum.accumulate(np.where(is_new, total_count, 0))
        cum_count_s: np.typing.ArrayLike = np.clip(total_count - group_offsets + 1, 1, anchors_count)

        cum_count: np.typing.ArrayLike = np.empty_like(cum_count_s)
        cum_count[sort_idx] = cum_count_s

        return cum_count


    def process_input_data(self, img_path: str) -> tf.Tensor:
        """ Removes alpha channel, rescales and normalize image.

        Args:
            img_path (str): path to training image.

        Returns:
            tf.Tensor: processed image.
        """
        img: tf.Tensor = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [360, 640])
        img = tf.cast(img, tf.float32) / 255.0
        
        return img


    def build_target_tensor(self, objects: tf.RaggedTensor) -> np.ndarray:
        """ Transforms bounding boxes into target tensors.

        Args:
            objects (np.ndarray): list of classed id and bounding boxes. 

        Returns:
            np.ndarray: target tensor [0:2] -> grid coordinates, [2:4] -> scaling factors twth, [4] -> objectness, [5:] -> classes.
        """
        objects_numpy: np.ndarray = objects.numpy()
        target_tensor: np.ndarray = np.zeros((*self.grid_size, len(self.anchors), 5 + self.num_classes), dtype=np.float32)

        grid_x: np.ndarray = objects_numpy[:,1]//self.stride
        grid_y: np.ndarray = objects_numpy[:,2]//self.stride

        x_offset: np.ndarray = objects_numpy[:,1]/self.stride - grid_x
        y_offset: np.ndarray = objects_numpy[:,2]/self.stride - grid_y

        target_params: np.ndarray = objects_numpy[:,-2:]
        iou_list: np.ndarray = iou_distances(target_params, self.anchors)

        anchors_ranking: np.ndarray = np.argsort(iou_list, axis=1)

        grid_repeat_count: np.ndarray = self.get_grid_count(np.stack([grid_y, grid_x]).T, len(self.anchors))

        best_anchor: np.ndarray = anchors_ranking[np.arange(anchors_ranking.shape[0]), anchors_ranking.shape[1]-grid_repeat_count]

        #  b_w = a_w * e^{t_w}
        #  t_w = log(b_w / a_w)
        target_tensor[grid_y, grid_x, best_anchor, 0:2] = np.array([y_offset, x_offset]).T
        target_tensor[grid_y, grid_x, best_anchor, 2:4] = np.array([np.log(np.array(objects_numpy)[:,4]/self.anchors[best_anchor][:,1]), np.log(np.array(objects_numpy)[:,3]/self.anchors[best_anchor][:,0])]).T
        target_tensor[grid_y, grid_x, best_anchor, 4] = 1.0
        target_tensor[grid_y, grid_x, best_anchor, 5 + np.array(objects_numpy)[:,0]] = 1.0
                
        return target_tensor


    def get_dataset(self, img_paths: np.ndarray , bbox_list: list , batch_size: int=32) -> tf.data.Dataset:
        """ Creates a TensorFlow dataset for object detection training by processing input
            images and generating corresponding YOLO target tensors from bounding boxes.
            The input image paths are converted into a dataset and processed in parallel.

        Args:
            img_paths (np.ndarray): image paths.
            bbox_list (list): bounding box lists.
            batch_size (int, optional): batch size. Defaults to 32.

        Returns:
            tf.data.Dataset: tuples of processed image tensor and yolo target tensor
        """
        dataset_imput = tf.data.Dataset.from_tensor_slices(img_paths)
        dataset_imput = dataset_imput.map(self.process_input_data, num_parallel_calls=tf.data.AUTOTUNE)

        def wrap_build_target(x):
            result = tf.py_function(func=self.build_target_tensor, inp=[x], Tout=tf.float32)
            
            result.set_shape((*self.grid_size, len(self.anchors), 5 + self.num_classes)) 
            return result

        dataset_output = tf.data.Dataset.from_tensor_slices(tf.ragged.constant(bbox_list))
        dataset_output = dataset_output.map(wrap_build_target , num_parallel_calls=tf.data.AUTOTUNE)

        dataset = tf.data.Dataset.zip((dataset_imput, dataset_output))
        dataset = dataset.shuffle(len(img_paths))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset


    def build_nn(self) -> Model:
        """ Creates yolo neural network.

        Returns:
            Model: tensorflow Model object.
        """
        def convolutional_block(x: tf.Tensor, filters: int, kernel_size: int, strides: int=1) -> tf.Tensor:

            x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.LeakyReLU(negative_slope=0.1)(x)

            return x
        
        inputs: tf.Tensor = layers.Input(shape=self.input_shape)

        x = convolutional_block(inputs, 16, 3, strides=5)
        
        x = convolutional_block(x, 32, 3, strides=2)
        
        x = convolutional_block(x, 64, 3, strides=2)
        
        x = convolutional_block(x, 128, 3, strides=2)

        x = convolutional_block(x, 256, 3)
        x = convolutional_block(x, 128, 1) 
        x = convolutional_block(x, 256, 3)

        output_filters: int = len(self.anchors) * (5 + self.num_classes)
        
        outputs: tf.Tensor = layers.Conv2D(output_filters, 1, activation='linear', name='yolo_output')(x)
        outputs = layers.Reshape((*self.grid_size, len(self.anchors), 5 + self.num_classes))(outputs) 

        model = Model(inputs, outputs)

        return model
    

    def train_model(self, val_split: float=0.2, save_model: bool=False, model_name: str = 'model_yolo') -> Model:
        """ Trains yolo neurla network with created train and validation set.

        Args:
            val_split (float, optional): validation set split size. Defaults to 0.2.
            save_model (bool, optional): save model flag. Defaults to False.
            model_name (str, optional): name of the model. Defaults to 'model_yolo'.

        Returns:
            Model: trained tensorflow Model object.
        """
        model: Model = self.build_nn()
        yolo_loss = YoloLoss()
        # model.summary(line_length = 100)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=yolo_loss
        )

        with open(os.path.join(self.training_set_dir, "Targets.pickle"), "rb") as input_file:
            targets: list = pickle.load(input_file)

        last_idx_train: int = int(len(targets) * (1-val_split))
        last_idx_val: int = len(targets)
            
        img_paths = np.array([os.path.join(self.training_set_dir, f"image_{i}.jpg") for i in range(last_idx_train)], dtype=str)
        bbox_list = [bbox["objects"] for bbox in targets[:last_idx_train]]
        dataset = self.get_dataset(img_paths, bbox_list)

        img_paths_val = np.array([os.path.join(self.training_set_dir, f"image_{i}.jpg") for i in range(last_idx_train, last_idx_val)], dtype=str)
        bbox_list_val = [bbox["objects"] for bbox in targets[last_idx_train:]]
        dataset_val = self.get_dataset(img_paths_val, bbox_list_val)
    
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, start_from_epoch=5, restore_best_weights=True, verbose=1)
        # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=2, min_lr=1e-7, verbose=1)
        def lr_schedule(epoch, lr):
            if epoch < 2:
                return lr 
            else:
                return 0.0001
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

        history = model.fit(dataset, epochs=50, validation_data=dataset_val, verbose=1, callbacks=[lr_scheduler, callback])

        if save_model:
            model.save(os.path.join(self.models_dir, f"{model_name}.keras"))

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig(os.path.join(self.models_dir, f"{model_name}.png"))
        plt.show()

        return model
    

          
if __name__ == "__main__":
    ag = AnchorsGenerator()
    anchors = ag.anchor_kmeans(5, 100)

    ta = TrainingAssistant(anchors)
    model = ta.train_model(save_model=False)
