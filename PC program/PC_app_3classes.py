import cv2
import numpy as np
    
class CameraWindow():
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_name = ''
        self.image_height = 0
        self.image_width = 0
        self.labels = ['correct_mask', 'unmasked', 'incorrect_mask']
    
    def image_shape(self, image):
        return image.shape[0], image.shape[1]
    
    def colors_init(self):
        colors = ['0,255,0','255,0,0','0,0,255']
        colors = [np.array(every_color.split(',')).astype('int') for every_color in colors]
        colors = np.array(colors)
        colors = np.tile(colors,(16,1))
        
        return colors
    
    def model_output_layer(self, model):
        model_layers = model.getLayerNames()
        model_output_layer = [model_layers[yolo_layer - 1] for yolo_layer in model.getUnconnectedOutLayers()]
        
        return model_output_layer
    
    def bounding_box_coordinates(self, object_detection):
        bounding_box = object_detection[0:4] * np.array([self.image_width, self.image_height, self.image_width, self.image_height])
        (x, y, box_width, box_height) = bounding_box.astype("int")
        box_x = int(x - (box_width / 2))
        box_y = int(y - (box_height / 2))
        
        return box_x, box_y, box_width, box_height
        
    def nms_update(self, predicted_class_id, prediction_confidence, object_detection, nms_classes, nms_confidences, nms_boxes):
        box_x, box_y, box_width, box_height = self.bounding_box_coordinates(object_detection)
        
        nms_classes.append(predicted_class_id)
        nms_confidences.append(float(prediction_confidence))
        nms_boxes.append([box_x, box_y, int(box_width), int(box_height)])
        
        return nms_classes, nms_confidences, nms_boxes
    
    def non_max_supression(self, model_detection_layers):
        nms_classes = []
        nms_confidences = []
        nms_boxes = []
        for object_detection_layer in model_detection_layers:
            for object_detection in object_detection_layer:
                scores = object_detection[5:]
                predicted_class_id = np.argmax(scores)
                prediction_confidence = scores[predicted_class_id]
                
                if prediction_confidence > 0.50:
                    nms_classes, nms_confidences, nms_boxes = self.nms_update(predicted_class_id, prediction_confidence, object_detection, nms_classes, nms_confidences, nms_boxes)
                    
        return nms_classes, nms_confidences, nms_boxes
    
    def get_predicted_class_id(self, best_nms_class_score, nms_classes):
        predicted_class_id = nms_classes[best_nms_class_score]
        return predicted_class_id
    
    def get_predicted_class_label(self, predicted_class_id):
        predicted_class_label = self.labels[predicted_class_id]
        return predicted_class_label
    
    def get_prediction_confidence(self, best_nms_class_score, nms_confidences):
        prediction_confidence = nms_confidences[best_nms_class_score]
        return prediction_confidence
    
    def bbox_coords(self, coord_dot, lenght):
        return coord_dot + lenght
    
    def draw_rectangle(self, image, box_x, box_y, box_x_end, box_y_end, box_color):
        cv2.rectangle(image, (box_x, box_y), (box_x_end, box_y_end), box_color, 5)
        
    def put_text(self, image, predicted_class_label, box_x, box_y, box_color):
        cv2.putText(image, predicted_class_label, (box_x, box_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    
    def draw_bbox(self, best_nms_score, nms_classes, nms_confidences, nms_boxes, image, colors):
        for max_valueid in best_nms_score:
            try:
                best_nms_class_score = max_valueid
            except:
                print(f"max_value {max_valueid}")
            box = nms_boxes[best_nms_class_score]
            box_x = box[0]
            box_y = box[1]
            box_width = box[2]
            box_height = box[3]
            
            predicted_class_id = self.get_predicted_class_id(best_nms_class_score, nms_classes)
            predicted_class_label = self.get_predicted_class_label(predicted_class_id)
            prediction_confidence = self.get_prediction_confidence(best_nms_class_score, nms_confidences)
        
            box_x_end = self.bbox_coords(box_x, box_width)
            box_y_end = self.bbox_coords(box_y, box_height)
        
            box_color = colors[predicted_class_id]
        
            box_color = [int(c) for c in box_color]
        
            predicted_class_label = f'{predicted_class_label}: {round(prediction_confidence * 100, 2)}'
            
            self.draw_rectangle(image, box_x, box_y, box_x_end, box_y_end, box_color)
            self.put_text(image, predicted_class_label, box_x, box_y, box_color)
                            
    def yolov4_model(self):
        
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        while True:
            ret, frame = cap.read()
            
            self.image_height, self.image_width = self.image_shape(frame)

            blob = cv2.dnn.blobFromImage(frame, 0.003922, (416, 416), crop=False)

            colors = self.colors_init()

            model = cv2.dnn.readNetFromDarknet('yolov4.cfg','yolov4_best.weights')
            
            model_output_layer = self.model_output_layer(model)
            model.setInput(blob)
            model_detection_layers = model.forward(model_output_layer)
            
            nms_classes = []
            nms_boxes = []
            nms_confidences = []
            
            nms_classes, nms_confidences, nms_boxes = self.non_max_supression(model_detection_layers)     
            
            best_nms_score = cv2.dnn.NMSBoxes(nms_boxes, nms_confidences, 0.5, 0.4)

            self.draw_bbox(best_nms_score, nms_classes, nms_confidences, nms_boxes, frame, colors)
            
            cv2.imshow('Input', frame)
            c = cv2.waitKey(1)
            if c == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
            

if __name__ == '__main__':
    CameraWindow().yolov4_model()