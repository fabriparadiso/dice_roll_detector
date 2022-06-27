import cv2
import numpy as np

class Video():
    def __init__(self, video) -> None:
        self.frames_rgb = []
        self.height = None
        self.n_frames = None
        self.video = cv2.VideoCapture(video)
        self.width = None

    def create_new_video(self,
                         numbers: np.array,
                         initial_steady_position: int,
                         final_steady_position: int,
                         heights: np.array,
                         widths: np.array,
                         x: np.array,
                         y: np.array) -> None:

        for i in range(initial_steady_position, final_steady_position+1):
            for j in range (0, len(widths)):
                cv2.rectangle(self.frames_rgb[i], (x[j] + widths[j], y[j] + heights[j]), (x[j], y[j]), (0, 0, 255), 3)
                cv2.putText(self.frames_rgb[i], str(numbers[j]) , (x[j], y[j]), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
                if(i==final_steady_position):
                    detection_rgb = cv2.cvtColor(self.frames_rgb[i], cv2.COLOR_BGR2RGB)
                    cv2.imwrite("detection.png", detection_rgb)

        new_video = cv2.VideoWriter('dice_roll_detected.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (self.width, self.height))
        for i in range(0, self.n_frames-1):
            new_video.write(cv2.cvtColor(self.frames_rgb[i], cv2.COLOR_RGB2BGR))
        new_video.release()

    def get_video_information(self) -> None:
        self.n_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

    def store_video_frames(self) -> None:
        i = 0
        while (self.video.isOpened()):
            i = i + 1
            _, frame = self.video.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frames_rgb.append(frame_rgb)
            if i == self.n_frames - 1:
                break

def are_dices_steady(all_dice_counted: bool, centx: np.array, centx_old: np.array, centy: np.array, centy_old: np.array, store_old_centroids: bool) -> bool:
    are_dices_steady_flag = False
    if(all_dice_counted is True):
            if(store_old_centroids is True):
                diffx = np.subtract(centx, centx_old)
                diffy = np.subtract(centy, centy_old)
                if(np.count_nonzero(diffx) < 3 and np.count_nonzero(diffy) < 3):
                    are_dices_steady_flag = True
    return are_dices_steady_flag, centx_old, centy_old

def binaryze_rgb_image_with_blue_plane_using_mask(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _, _, blue = cv2.split(frame)
    filtered_image = cv2.bitwise_and(blue, mask)
    _, points = cv2.threshold(filtered_image, 120, 255, cv2.THRESH_BINARY)
    return points

def calculate_centroids(x: int, y: int, height: int) -> bool:
    return x + height/2, y + height/2

def closure_with_ellipse_kernel(mask: np.ndarray, size: tuple) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

def count_dices_centroids(contours_dices: np.ndarray, number_of_dices: int = 5) -> tuple:
    all_dice_counted = False
    centx = []
    centy = []
    dice_counter = 0
    for contour in contours_dices:
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = float(width)/height
        _ , radius = cv2.minEnclosingCircle(contour)
        if(detecting_dice(aspect_ratio, radius) is True):
            cx, cy = calculate_centroids(x, y, height)
            centx.append(cx)
            centy.append(cy)
            dice_counter+=1
        if (dice_counter==number_of_dices):
            all_dice_counted = True

    return all_dice_counted, centx, centy

def counting_dice(frames_rgb: np.ndarray) -> tuple:
    initial_steady_position, final_steady_position, mask_dices = get_steady_frame_indexes(frames_rgb)
    fill_outline_inside_mask(mask_dices[final_steady_position])
    points = binaryze_rgb_image_with_blue_plane_using_mask(frames_rgb[final_steady_position], mask_dices[final_steady_position])
    final_points = erode_with_ellipse_kernel(points, (3,3))
    mask_closure = closure_with_ellipse_kernel(mask_dices[final_steady_position], (7,7))
    binary_dices = cv2.absdiff(mask_closure, final_points)
    binary_dices_closure = closure_with_ellipse_kernel(binary_dices, (3,3))
    dices_numbers, X, Y, HEIGHTS, WIDTHS = get_dices_numbers(binary_dices_closure)
    return initial_steady_position, dices_numbers, final_steady_position, X, Y, HEIGHTS, WIDTHS

def detecting_dice(aspect_ratio: float,
                   radius: float,
                   MAX_ASPECT_RATIO: float = 1.05,
                   MIN_ASPECT_RATIO: float = 0.6,
                   MAX_RADIUS_ENCLOSING_CIRCLE: int = 50,
                   MIN_RADIUS_ENCLOSING_CIRCLE: int = 28
                   ) -> bool:
    is_dice = False,
    if(aspect_ratio > MIN_ASPECT_RATIO and aspect_ratio < MAX_ASPECT_RATIO and radius > MIN_RADIUS_ENCLOSING_CIRCLE and radius < MAX_RADIUS_ENCLOSING_CIRCLE):
        is_dice = True
    return is_dice

def erode_with_ellipse_kernel(mask: np.ndarray, size: tuple) -> np.ndarray:
    kernel = cv2.getStructuringElement(shape = cv2.MORPH_ELLIPSE, ksize = size)
    return cv2.erode(mask, kernel)

def fill_outline_inside_mask(binary_image: np.ndarray) -> None:
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(binary_image, [contour], -1, color=255, thickness=-1)

def get_steady_frame_indexes(frames_rgb: np.ndarray) -> tuple:
    initial_index_steady_frame = None
    centx_old = []
    centy_old = []
    final_index_steady_frame = None
    mask_dices = []
    steady_start = False
    store_old_centroids = False
    for index, frame in enumerate(frames_rgb):
        plane_red , _, _ = cv2.split(frame)
        mask_red = cv2.inRange(plane_red, 50, 255)
        dices_bw = erode_with_ellipse_kernel(mask_red, (6,6))
        mask_dices.append(dices_bw)
        contours_dices, _ = cv2.findContours (dices_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        all_dice_counted, centx, centy = count_dices_centroids(contours_dices)
        are_dices_steady_flag, centx_old, centy_old = are_dices_steady(all_dice_counted, centx, centx_old, centy, centy_old, store_old_centroids)
        if (all_dice_counted is True and steady_start is False):
            centx_old = centx.copy()
            centy_old = centy.copy()
            store_old_centroids = True
        if (are_dices_steady_flag is True and steady_start is False):
            initial_index_steady_frame = index
            steady_start = True
        if (are_dices_steady_flag is True and steady_start is True):
            final_index_steady_frame = index
    return initial_index_steady_frame, final_index_steady_frame, mask_dices

def get_dice_number(dice_binary_image: np.ndarray) -> int:
    dice_contour, _ = cv2.findContours(dice_binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in dice_contour:
        if(cv2.contourArea(contour) < 50):
            cv2.drawContours (dice_binary_image, [contour], -1, color = 255, thickness = -1)
    all_contour_dice, _ = cv2.findContours(dice_binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    number = len(all_contour_dice)-1
    return number

def get_dices_numbers(binary_dices_image: np.ndarray) -> tuple:
    numbers = []
    X = []
    Y = []
    WIDTHS = []
    HEIGHTS = []
    contours, _ = cv2.findContours(binary_dices_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = float(width)/height
        if(area < 6000 and area > 3200 and aspect_ratio < 1.1):
            dice = binary_dices_image[y:y + height, x:x + width]
            dice_erode = erode_with_ellipse_kernel(dice, (4,4))
            dice_closure = closure_with_ellipse_kernel(dice_erode, (5,5))
            numbers.append(get_dice_number(dice_closure))
            X.append(x)
            Y.append(y)
            WIDTHS.append(width)
            HEIGHTS.append(height)
    return numbers, X, Y, WIDTHS, HEIGHTS


def main():
    video: Video = Video('dice_roll.mp4')
    video.get_video_information()
    video.store_video_frames()
    initial_steady_position, dices_numbers, final_steady_position, x, y, heights, widths = counting_dice(video.frames_rgb)
    print(dices_numbers)
    video.create_new_video(dices_numbers, initial_steady_position, final_steady_position, heights, widths, x, y)
if __name__ == "__main__":
    main()