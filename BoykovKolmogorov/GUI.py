import cv2


class GUI_seeds():

    def __init__(self, img_path):
        self.img_path = img_path
        self.drawing = False
        self.mode = "ob"
        self.marked_ob_pixels = []
        self.marked_bg_pixels = []
        self.image = cv2.imread(img_path)

    def mark_seeds(self, event, x, y, flags, param):
        h, w, _ = self.image.shape

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                if self.mode == "ob":
                    if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
                        self.marked_ob_pixels.append((y,x))
                    cv2.circle(self.image, (x,y), 2, (0,0,255), 2)
                else:
                    if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
                        self.marked_bg_pixels.append((y,x))
                    cv2.circle(self.image, (x,y), 2, (255,0,0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.mode == "ob":
                cv2.line(self.image,(x-3,y),(x+3,y),(0,0,255))
            else:
                cv2.line(self.image,(x-3,y),(x+3,y),(255,0,0))

    def labelling(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', self.mark_seeds)
        while(1):
            cv2.imshow('image',self.image)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('b'):
                self.mode = "bg"
            elif k == ord('o'):
                self.mode = "ob"
            elif k == 27:
                break
        cv2.destroyAllWindows()
        return self.marked_ob_pixels, self.marked_bg_pixels