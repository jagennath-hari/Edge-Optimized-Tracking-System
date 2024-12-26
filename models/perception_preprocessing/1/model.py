from torch.utils.dlpack import from_dlpack, to_dlpack
import torch
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        """Called when the model is being initialized."""
        pass

    def execute(self, requests):
        """Called when inference is requested."""
        responses = []

        for request in requests:
            # Retrieve the input tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, "input_image")
            image = from_dlpack(input_tensor.to_dlpack())  # Convert to PyTorch tensor

            if not image.is_cuda:
                image = image.to("cuda:0")  # Ensure the image is on GPU

            # Validate input shape
            if image.dim() != 3 or image.shape[2] != 3:
                raise ValueError("Input must be a 3D tensor with 3 channels (H, W, C).")

            original_height, original_width = image.shape[:2]

            # Create original dimensions as GPU tensors
            original_height_tensor = torch.tensor([original_height], device="cuda:0", dtype=torch.int32)
            original_width_tensor = torch.tensor([original_width], device="cuda:0", dtype=torch.int32)

            # Preprocess the image
            preprocessed_image = self.preprocess_image(image)

            # Convert outputs to Triton-compatible tensors
            preprocessed_blob = pb_utils.Tensor.from_dlpack("preprocessed_blob", to_dlpack(preprocessed_image))
            original_height_tensor = pb_utils.Tensor.from_dlpack("original_height", to_dlpack(original_height_tensor))
            original_width_tensor = pb_utils.Tensor.from_dlpack("original_width", to_dlpack(original_width_tensor))

            # Create the inference response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[preprocessed_blob, original_height_tensor, original_width_tensor]
            )
            responses.append(inference_response)

        return responses

    def preprocess_image(self, image):
        """Preprocess the input image."""
        # Convert BGR to RGB, normalize to [0, 1], and convert to FP16
        image = image[..., [2, 1, 0]].permute(2, 0, 1).to(dtype=torch.float16) / 255.0

        # Resize to 736x736
        resized_image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(736, 736), mode="bilinear", align_corners=False
        )

        return resized_image  # Shape: [1, 3, 736, 736]

    def finalize(self):
        """Called during model cleanup."""
        pass
