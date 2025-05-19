import cv2
import numpy as np

def find_hough_circles(image, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold, post_process=True, top_k=20):
    img_height, img_width = edge_image.shape[:2]
    dtheta = int(360 / num_thetas)
    thetas = np.arange(0, 360, step=dtheta)
    rs = np.arange(r_min, r_max, step=delta_r)
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    circle_candidates = [(r, int(r * cos_thetas[t]), int(r * sin_thetas[t]))
                         for r in rs for t in range(num_thetas)]

    from collections import defaultdict
    accumulator = defaultdict(int)

    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y, x] != 0:
                for r, dx, dy in circle_candidates:
                    xc = x - dx
                    yc = y - dy
                    if 0 <= xc < img_width and 0 <= yc < img_height:
                        accumulator[(xc, yc, r)] += 1

    output_img = image.copy()
    all_circles = [
        (x, y, r, votes / num_thetas)
        for (x, y, r), votes in accumulator.items()
        if votes / num_thetas >= bin_threshold
    ]

    # Sort by vote strength
    all_circles.sort(key=lambda c: -c[3])
    if top_k:
        all_circles = all_circles[:top_k]

    # Post-process: remove near-duplicates
    if post_process:
        final_circles = []
        for x, y, r, v in all_circles:
            too_close = False
            for xc, yc, rc, _ in final_circles:
                if np.hypot(x - xc, y - yc) < 10 and abs(r - rc) < 5:
                    too_close = True
                    break
            if not too_close:
                final_circles.append((x, y, r, v))
        all_circles = final_circles

    for x, y, r, _ in all_circles:
        cv2.circle(output_img, (x, y), r, (0, 255, 0), 1)  # thin stroke for circles

    return output_img, all_circles


def main():
    r_min = 100
    r_max = 200
    delta_r = 1
    num_thetas = 100
    bin_threshold = 0.4
    min_edge_threshold = 100
    max_edge_threshold = 200
    
    input_img = cv2.imread("sobel_small.png")
    
    # Edge detection on the input image
    edge_image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)
    
    if edge_image is not None:
        
        print ("Detecting Hough Circles Started!")
        circle_img, circles = find_hough_circles(input_img, edge_image, r_min, r_max, delta_r, num_thetas, bin_threshold)
        
        if circle_img is not None:
            cv2.imwrite("circles_img.png", circle_img)
    else:
        print ("Error in input image!")
            
    print ("Detecting Hough Circles Complete!")



if __name__ == "__main__":
    main()