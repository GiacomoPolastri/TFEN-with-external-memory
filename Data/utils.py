from detectron2.structures import BoxMode

def get_fiftyone_dicts(samples):
    
    samples.compute_metadata()
    dataset_dicts = []
    for sample in samples.select_fields(["id", "filepath", "metadata", "detections"]):
    
        height = sample.metadata["height"]
        width = sample.metadata["width"]
        record = {}
        record["file_name"] = sample.filepath
        record["image_id"] = sample.id
        record["height"] = height
        record["width"] = width

        objs = []
        for det in sample.detections.detections:
            tlx, tly, w, h = det.bounding_box
            bbox = [int(tlx*width), int(tly*height), int(w*width), int(h*height)]
            fo_poly = det.to_polyline()
            poly = [(x*width, y*height) for x, y in fo_poly.points[0]]
            poly = [p for x in poly for p in x]
            obj = {
                "bbox" : bbox,
                "bbox_mode" : BoxMode.XYWH_ABS,
                # Before detections, but the default structure says segmentation
                "segmentation" : [poly],
                "category_id" : get_category_id(det.label),
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts

def get_category_id(label):
    category_id = 0
    if label == "boat": 
        category_id = 1
        return category_id
    if label == "car":
        category_id = 2
        return category_id
    if label == "dock":
        category_id = 3
        return category_id
    if label == "jetski":
        category_id = 4
        return category_id
    if label == "lift":
        return category_id