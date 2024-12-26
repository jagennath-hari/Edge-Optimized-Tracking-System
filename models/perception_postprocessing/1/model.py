from torch.utils.dlpack import from_dlpack, to_dlpack
import torch
import numpy as np
import triton_python_backend_utils as pb_utils
import torchvision.ops as ops

class TritonPythonModel:
    def initialize(self, args):
        """Called when the model is being initialized."""
        pass

    def execute(self, requests):
        """Perform class-aware NMS on raw model outputs."""
        responses = []

        for request in requests:
            # Retrieve inputs using DLPack
            output0 = from_dlpack(pb_utils.get_input_tensor_by_name(request, "output0").to_dlpack())
            original_width = pb_utils.get_input_tensor_by_name(request, "original_width").as_numpy()[0]
            original_height = pb_utils.get_input_tensor_by_name(request, "original_height").as_numpy()[0]

            # Ensure tensors are on GPU
            if not output0.is_cuda:
                output0 = output0.to("cuda:0")

            # Parse `output0`: [1, num_features, num_boxes]
            boxes = output0[0, :4, :].permute(1, 0)  # Shape: [num_boxes, 4] (center_x, center_y, width, height)
            confidences = output0[0, 4:, :]          # Shape: [num_classes, num_boxes]

            # Convert boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)
            boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = center_x - width / 2
            boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = center_y - height / 2
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]      # x2 = x1 + width
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]      # y2 = y1 + height

            # Identify the best class per box
            scores, class_ids = torch.max(confidences, dim=0)

            # Filter boxes with confidence > 0.5
            valid_indices = scores > 0.5
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            class_ids = class_ids[valid_indices]

            # Perform batched NMS
            keep_indices = ops.batched_nms(boxes, scores, class_ids, iou_threshold=0.7)

            # Filter outputs based on NMS
            boxes = boxes[keep_indices]
            scores = scores[keep_indices]
            class_ids = class_ids[keep_indices]

            # Scale boxes back to the original image size
            scale_x = original_width / 736.0
            scale_y = original_height / 736.0
            boxes[:, 0] *= scale_x  # x1
            boxes[:, 1] *= scale_y  # y1
            boxes[:, 2] *= scale_x  # x2
            boxes[:, 3] *= scale_y  # y2
            boxes = boxes.int()  # Convert to integer coordinates

            # Convert outputs to CPU for Triton response
            boxes_cpu = boxes.cpu().numpy()
            scores_cpu = scores.cpu().numpy()
            class_ids_cpu = class_ids.cpu().numpy()

            # Create output tensors
            boxes_tensor = pb_utils.Tensor("boxes", boxes_cpu.astype(np.int32))
            scores_tensor = pb_utils.Tensor("scores", scores_cpu.astype(np.float16))
            class_ids_tensor = pb_utils.Tensor("class_ids", class_ids_cpu.astype(np.int32))

            # Append the response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[boxes_tensor, scores_tensor, class_ids_tensor]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """Called during model cleanup."""
        pass
