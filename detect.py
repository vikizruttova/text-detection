import httpx
import asyncio
import aiofiles.os
from aiofiles import open as aio_open
import torch
from PIL import Image
from io import BytesIO
import os
import json
import cv2
import numpy as np
import time
from tqdm import tqdm
from mmocr.apis import MMOCRInferencer


image_directory = 'images_gerulata'
os.makedirs(image_directory, exist_ok=True)
model_names = {"dbnet", "maskrcnn", "psenet", "textsnake", "drrg", "fcenet"}
models = {}
for model_name in model_names:
    models[model_name] = MMOCRInferencer(det=model_name, device="cpu")

image_results = {}
async def fetch_image(image_url, image_filename):
    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)

        if response.status_code == 200:
            async with aio_open(image_filename, 'wb') as f:
                await f.write(response.content)

    return image_filename
def calculate_polygon_area(polygon):
    polygon = np.array(polygon).reshape(-1, 2)
    return 0.5 * abs(np.dot(polygon[:, 0], np.roll(polygon[:, 1], 1)) - np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))
def get_detection_results(fname, ocr):
    result = ocr(fname)
    image = cv2.imread(fname)
    image_area = image.shape[0] * image.shape[1]
    thresholds = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]

    for threshold in thresholds:
        covered_area = 0
        for polygon, confidence in zip(result['predictions'][0]['det_polygons'],
                                       result['predictions'][0]['det_scores']):
            if confidence < 0.75:
                continue
            area = calculate_polygon_area(polygon)
            covered_area += area

        if covered_area / image_area > threshold:
            return result, threshold

    return None, None
def process_image_and_save_results(image_filename, model_name, model_type, threshold):
    ocr = models[model_name]

    start_time = time.time()
    result, _ = get_detection_results(image_filename, ocr)

    # read again to draw the bboxes
    image = cv2.imread(image_filename)

    detection_flag = "negative"
    if result is not None:
        for polygon, confidence in zip(result['predictions'][0]['det_polygons'],
                                       result['predictions'][0]['det_scores']):
            if confidence < 0.75:
                continue
            polygon = np.array(polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [polygon], True, (0, 255, 0), 2)
            detection_flag = "positive"

    # use detection_flag to specify which folder to put the image in
    output_folder = f'output_folder/{model_name}/{model_type}/{threshold}/{detection_flag}'
    os.makedirs(output_folder, exist_ok=True)

    output_filename = os.path.join(output_folder, os.path.basename(image_filename))
    cv2.imwrite(output_filename, image)

    print(f"Detection result for {image_filename}: {detection_flag}")
    print(f"Execution time for {image_filename}: {time.time() - start_time} seconds")

    return result, detection_flag
async def process_image(image_data):
    storage_key = image_data['storage_key']
    image_url = f"https://img.gerulata.com/{storage_key}"

    image_filename = os.path.join(image_directory, f"{storage_key.split('/')[-1]}.jpg")
    await aiofiles.os.makedirs(os.path.dirname(image_filename), exist_ok=True)

    await fetch_image(image_url, image_filename)

    for model_name in model_names:
        model_type = image_data['model_type']
        thresholds = [0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15]

        for threshold in thresholds:
            # This function can also be made asynchronous for better performance
            result, detection_flag = process_image_and_save_results(image_filename, model_name, model_type, threshold)
            image_results[image_filename] = {"result": result, "detection": detection_flag}


async def main():
    with open('Result_10.json') as f:
        json_data = json.load(f)

    tasks = [process_image(image_data) for image_data in tqdm(json_data)]
    await asyncio.gather(*tasks)


# Run the asynchronous main function
asyncio.run(main())
