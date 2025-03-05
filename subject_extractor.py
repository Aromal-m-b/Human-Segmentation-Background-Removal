import cv2
import numpy as np
from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8s-seg.pt").to(device)# Use "yolov8n-seg.pt" for a lighter version

# Read image
image_path = "human.jpg"
image = cv2.imread(image_path)
original_image = image.copy()
image_with_detections = image.copy()

# Run inference
results = model(image)

# Find the human with the highest confidence
best_conf = 0
best_mask = None
best_box = None

for result in results:
    for i, mask in enumerate(result.masks.xy):
        conf = result.boxes.conf[i].cpu().numpy()  # Confidence score
        if conf > best_conf:
            best_conf = conf
            best_mask = mask
            best_box = result.boxes.xyxy[i].cpu().numpy().astype(int)  # Bounding box

# If a human is detected, process the highest confidence one
if best_mask is not None:
    # Create a binary mask
    mask_img = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_img, [np.array(best_mask, dtype=np.int32)], 255)

    # Draw detection results
    cv2.rectangle(image_with_detections, (best_box[0], best_box[1]), (best_box[2], best_box[3]), (0, 255, 0), 3)
    overlay = image.copy()
    overlay[:, :, 1] = np.where(mask_img == 255, 255, overlay[:, :, 1])  # Green overlay
    image_with_detections = cv2.addWeighted(image_with_detections, 1, overlay, 0.5, 0)

    # Function to resize image to fit screen
    def resize_for_display(img, max_width=800, max_height=600):
        h, w = img.shape[:2]
        scale = min(max_width / w, max_height / h)  # Scale factor
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # Function to apply selected operation
    def apply_operation(option):
        global displayed_image

        if option == "white_bg":
            white_bg = np.ones_like(original_image) * 255  # White image
            displayed_image = np.where(mask_img[..., None] == 255, original_image, white_bg)
        elif option == "transparent_bg":
            checker_size = 20  # Checkerboard square size
            h, w = original_image.shape[:2]
            checkerboard = np.zeros((h, w, 3), dtype=np.uint8)

            for i in range(0, h, checker_size):
                for j in range(0, w, checker_size):
                    if (i // checker_size + j // checker_size) % 2 == 0:
                        checkerboard[i:i+checker_size, j:j+checker_size] = (200, 200, 200)  # Light grey
                    else:
                        checkerboard[i:i+checker_size, j:j+checker_size] = (255, 255, 255)  # White

            displayed_image = np.where(mask_img[..., None] == 255, original_image, checkerboard)
        elif option == "detection":
            displayed_image = image_with_detections  # Show detected human
        elif option == "save":
            cv2.imwrite("output.jpg", displayed_image)
            print("Image saved as output.jpg")
            cv2.destroyAllWindows()
            exit()  # Exit after saving

        # Redraw the image with options
        update_display()

    # Function to update display with selected operation
    def update_display():
        img_with_options = displayed_image.copy()

        # Increase font size for better visibility
        font_scale = 1.2
        font_thickness = 3

        cv2.putText(img_with_options, "1) White Background", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
        cv2.putText(img_with_options, "2) Transparent Sticker", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
        cv2.putText(img_with_options, "3) Detection", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
        cv2.putText(img_with_options, "4) Save", (20, 600), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 8)
        cv2.putText(img_with_options, "Press ESC to Exit", (20, 750), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)


        resized_display = resize_for_display(img_with_options)
        cv2.imshow("Human Detection", resized_display)

    # Start with detected image and menu
    displayed_image = image_with_detections
    update_display()

    while True:
        key = cv2.waitKey(0)

        if key == ord("1"):  # White background
            apply_operation("white_bg")
        elif key == ord("2"):  # Transparent sticker
            apply_operation("transparent_bg")
        elif key == ord("3"):  # Show detection result
            apply_operation("detection")
        elif key == ord("4"):  # Save current image
            apply_operation("save")
        elif key == 27:  # ESC key to exit
            cv2.destroyAllWindows()
            quit()

else:
    print("No humans detected.")
