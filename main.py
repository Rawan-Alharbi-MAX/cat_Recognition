import cv2
from ultralytics import YOLO

# تحميل النموذج المدرب
model = YOLO("yolov8n.pt")
print("Loaded YOLOv8n model successfully!")
 # فتح فيديو 
cap = cv2.VideoCapture("video.mp4")  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # تمرير الفريم للنموذج
    results = model(frame)

    # استخراج الكائنات
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name == "cat":
                # رسم المربع حول القطة فقط
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # عرض الإطار
    cv2.imshow("Cat Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
