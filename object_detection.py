import tensorflow as tf
import logging
from object_detection.utils import config_util
#from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import numpy as np
import robot_control as control_robot


class ObjectDetection:

    def __init__(self,
                 robot=None,
                 robot_speed=20,
                 config_path='ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config',
                 checkpoint_path='my_model',
                 annotation_path='data'):

        logging.info("Loading pipeline config and building a detection model")
        self.configs = config_util.get_configs_from_pipeline_file(config_path)
        self.detection_model = model_builder.build(model_config=self.configs['model'], is_training=False)

        logging.info("Restoring checkpoint")
        self.ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        self.ckpt.restore(os.path.join(checkpoint_path, 'ckpt-6')).expect_partial()
        self.category_index = label_map_util.create_category_index_from_labelmap(annotation_path + '/label_map.pbtxt')

        logging.info("Setting up camera")
        self.camera = cv2.VideoCapture(0)
        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logging.info("Initializing robot")
        self.robot = robot

    def __enter__(self):
        """
        Entering a with statement
        """
        return self

    def __exit__(self, _type, value, traceback):
        """
        Exit a with statement
        """
        if traceback is not None:
            # If any exception occurred:
            logging.error('Exiting with statement with exception %s' % traceback)

        self.cleanup()

    @tf.function
    def detect_fn(self, image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections

    def print_class(self, class_id, score_id, category_index):
        if score_id >= 90:
            self.name_class = list(category_index.values())
            name_class_final = self.name_class[class_id]["name"]

            return name_class_final

    def detect_objects(self):
        while self.camera.isOpened():
            _, frame = self.camera.read()
            image_np = np.array(frame)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = self.detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                self.category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.6,
                agnostic_mode=False)

            class_final = detections['detection_classes'][0]
            score_final = (detections['detection_scores'][0]) * 100

            detected_object = self.print_class(class_final, score_final, self.category_index)
            print(detected_object)
            if self.robot is not None:
                # maybe from here control the robot based on the class
                if detected_object == 'go':
                    control_robot.go_straight_with_speed(20)
                else:
                    if detected_object == 'turn right':
                        control_robot.turn_right(1)
                    elif detected_object == 'turn left':
                        control_robot.turn_left(1)
                    elif detected_object == 'person':
                        control_robot.go_straight_with_speed(0)
                    elif detected_object == '20':
                        control_robot.go_straight_with_speed(20)
                    elif detected_object == '40':
                        control_robot.go_straight_with_speed(40)
                    elif detected_object == 'stop':
                        control_robot.stop()
                    else:
                        logging.info("Nothing detected. Waiting for command")
                        control_robot.go_straight_with_speed(0)

            return image_np_with_detections

            #cv2.imshow('object detection', image_np_with_detections)

    def cleanup(self):
        self.camera.release()
        cv2.destroyAllWindows()


def test_camera():

    objects_process = ObjectDetection()
    while True:
        objects_detected = objects_process.detect_objects()
        cv2.imshow('object detection', objects_detected)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':

    test_camera()
