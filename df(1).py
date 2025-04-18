from roboflowoak import RoboflowOak
import cv2
import time
import numpy as np

if __name__ == '__main__':
    # Specify OAK-D-Lite device explicitly
    rf = RoboflowOak(model="dead-fish-ye77u", confidence=0.05, overlap=0.5,
    version="30", api_key="yWnw1e55bLrQLj1LOs1H", rgb=True,
    depth=True, device="OAK-D-Lite", blocking=True)
    print("Model loaded")
    while True:
        t0 = time.time()
        
        # Run model inference
        result, frame, raw_frame, depth = rf.detect()
        predictions = result["predictions"]
        for prediction in result["predictions"]:
            confidence = prediction.confidence
            if confidence >= 0.80:  # Filter out predictions with confidence < 0.80
                x = int(prediction.x)
                y = int(prediction.y)
                w = int(prediction.width)
                h = int(prediction.height)
                label = prediction.class_name
        
        # Benchmarking
                t = time.time() - t0
                print("INFERENCE TIME IN MS ", 1/t)
                print("PREDICTIONS ", [p.json() for p in predictions])
                cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (255, 0, 0), 2)
                # Add label and confidence
                cv2.putText(frame, f'{label} {confidence:.2f}', (x - w // 2, y - h // 2 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Adjust depth visualization for OAK-D-Lite
        max_depth = np.amax(depth)
        if max_depth > 0:
            depth_normalized = depth / max_depth
        else:
            depth_normalized = depth
        
        cv2.imshow("depth", depth_normalized)
        cv2.imshow("frame", frame)

        # Stop inference on 'q' key
        if cv2.waitKey(1) == ord('q'):
            break
