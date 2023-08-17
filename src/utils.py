import pickle
import cv2
import os

def serialize(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

def deserialize(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)    

def convert_video2imgs(video_read_path, every_frame, img_save_path):
    cam = cv2.VideoCapture(video_read_path)
    curr_frame = 0
    print("Saving video as images: sampling every " + str(every_frame) + " frames")
    while(True):
        ret,frame = cam.read()    
        if ret: # if video frames remain
            if (curr_frame % every_frame) == 0:
                save_name = 'frame' + str(curr_frame) + '.jpg'
                save_path = os.path.join(img_save_path, save_name)
                print('Creating...' + save_name)
                cv2.imwrite(save_path, frame)
            curr_frame += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()
