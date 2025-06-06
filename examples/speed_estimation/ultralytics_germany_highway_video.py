import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

import supervision as sv

from supervision.geometry.core import Point
from supervision import Color

SOURCE = np.array([[649, 248], [1170, 267], [2953, 1079], [-1582, 1079]])

TARGET_WIDTH = 33
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed Estimation using Ultralytics and Supervision"
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to the source video file",
        type=str,
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to the target video file (output)",
        type=str,
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.3,
        help="Confidence threshold for the model",
        type=float,
    )
    parser.add_argument(
        "--iou_threshold", default=0.7, help="IOU threshold for the model", type=float
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(video_path=args.source_video_path)
    model = YOLO("yolov8x.pt")

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
    )

    frame_generator = sv.get_video_frames_generator(source_path=args.source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=args.iou_threshold)
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)

            labels = []
            label_colors = []

            for idx, (tracker_id, [_, y]) in enumerate(zip(detections.tracker_id, points)):
                coordinates[tracker_id].append(y)

            for idx, tracker_id in enumerate(detections.tracker_id):
                class_id = detections.class_id[idx]
                class_name = model.names[class_id]

                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    label = f"#{tracker_id} {class_name}"
                    labels.append(label)
                    label_colors.append((0, 255, 0)) 
                else:
                    y_start = coordinates[tracker_id][-1]
                    y_end = coordinates[tracker_id][0]
                    distance = abs(y_start - y_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    label = f"#{tracker_id} {class_name} {int(speed)} km/h"
                    labels.append(label)

                    if speed > 120:
                        label_colors.append((0, 0, 255)) 
                    else:
                        label_colors.append((0, 255, 0)) 

            annotated_frame = frame.copy()
            annotated_frame = sv.draw_polygon(annotated_frame, polygon=SOURCE, color=sv.Color.RED)
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            anchor_points = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

            for label, color, point in zip(labels, label_colors, anchor_points):
                annotated_frame = sv.draw_text(
                    scene=annotated_frame,
                    text=label,
                    text_anchor=Point(x=int(point[0]), y=int(point[1]) - 10), 
                    text_color=Color(*color), 
                    text_scale=text_scale,
                    text_thickness=thickness,
                )

            sink.write_frame(annotated_frame)
            cv2.imshow("frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
