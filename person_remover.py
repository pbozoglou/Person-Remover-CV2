import argparse
import cv2
import functions as fc


def video():
    video_path = opt.i

    model, _ = fc.yolov5_initialize()

    cap = cv2.VideoCapture(video_path)
    cap_fps = round(cap.get(cv2.CAP_PROP_FPS))

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('filename.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, (frame_width,frame_height))

    while(True):
        ret,frame = cap.read()

        if not ret:break

        before = frame.copy()
        before = cv2.resize(before, None, fx=0.3, fy=0.3)

        temp = cv2.cvtColor(before.copy(), cv2.COLOR_BGR2RGB)
        
        box_list, _, categories_list = fc.yolov5_predict(temp, model)
        box  = [box_list[j] for j,i in enumerate(categories_list) if i==0]
        
        after = fc.person_remover(temp,box)
        after = cv2.cvtColor(after, cv2.COLOR_RGB2BGR)

        cv2.imshow('After', after)
        out.write(after)

        if cv2.waitKey(1) == ord('q'):
            break

    
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='video path', required=True)  # file path or 0 for webcam
    opt = parser.parse_args()
    
    video()