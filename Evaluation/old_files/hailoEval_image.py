from typing import List, Generator, Optional, Tuple, Dict
from hailo_platform import (HEF, VDevice,
                            FormatType, HailoSchedulingAlgorithm)
from functools import partial
import numpy as np
from typing import List, Generator, Optional, Tuple, Dict
from pathlib import Path
from functools import partial
import queue
from loguru import logger
import numpy as np
from PIL import Image
from hailo_platform import (HEF, VDevice,
                            FormatType, HailoSchedulingAlgorithm)
IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')
from hailoEval_utils import RestOfGraph
from utils import transform,_convert_image_to_rgb
from time import time


class CLIPHailoInference:
    """
    Initialises target on creation means the resourcess are acquierd on creation of object.
    """
    def __init__(
        self, hef_path: str,preprocess_Path, batch_size: int = 1,
        input_type: Optional[str] = None, output_type: Optional[Dict[str, str]] = None,
        send_original_frame: bool = False,is_data_batched:bool = False) -> None:

        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        params = VDevice.create_params()    
        # Set the scheduling algorithm to round-robin to activate the scheduler
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

        self.hef = HEF(hef_path)
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(batch_size)      
        if input_type is not None:
            self._set_input_type(input_type)
        if output_type is not None:
            self._set_output_type(output_type)

        self.output_type = output_type
        self.send_original_frame = send_original_frame
        self.is_data_batched = is_data_batched
        
        self.preprocess = transform(self.get_input_shape()[0])
        if preprocess_Path != None:
            self.postprocess = RestOfGraph(preprocess_Path)

    def loadPostprocess(self,postprocessjson_path):
        post_process = RestOfGraph(postprocessjson_path)
        self.postprocess = post_process
    
    def encode_image(self,image):
        """
        image: image as np array
        """
        image = self.performPreprocess(image).unsqueeze(0).numpy()
        image = np.transpose(image, (0,2, 3, 1)).astype(np.float32,order='F')
        image = np.expand_dims(image.flatten(), axis=0)
        self.input_queue.put(image)
        self.run()
        result = self.output_queue.get()[1] # output is from queue is (input,ouput)
        imageEncoding = self.performPostprocess(result)
        return imageEncoding
    
    def onlyHEF(self,image):
        """
        Give data direct to model without preprocess
        """
        self.input_queue.put(image)
        self.run()
        result = self.output_queue.get()[1]
        return result
    
    def performPreprocess(self,input):
        """
        Like preprocess from:
        model,preprocess = clip.load(model)
        """
        return self.preprocess(input)
    
    def performPostprocess(self,input):
        """
        calculation of the cut off graph
        """
        return self.postprocess(input)
    
    def callback(
        self, completion_info, bindings_list: list, input_batch: list,
    ) -> None:
        """
        Callback function for handling inference results.

        Args:
            completion_info: Information about the completion of the 
                             inference task.
            bindings_list (list): List of binding objects containing input 
                                  and output buffers.
            processed_batch (list): The processed batch of images.
        """
        if completion_info.exception:
            logger.error(f'Inference error: {completion_info.exception}')
        else:
            for i, bindings in enumerate(bindings_list):
                # If the model has a single output, return the output buffer. 
                # Else, return a dictionary of output buffers, where the keys are the output names.
                if len(bindings._output_names) == 1:
                    result = bindings.output().get_buffer()
                else:
                    result = {
                        name: np.expand_dims(
                            bindings.output(name).get_buffer(), axis=0
                        )
                        for name in bindings._output_names
                    }
                self.output_queue.put((input_batch[i], result))

    def get_vstream_info(self) -> Tuple[list, list]:

        """
        Get information about input and output stream layers.

        Returns:
            Tuple[list, list]: List of input stream layer information, List of 
                               output stream layer information.
        """
        return (
            self.hef.get_input_vstream_infos(), 
            self.hef.get_output_vstream_infos()
        )

    def get_hef(self) -> HEF:
        """
        Get the object's HEF file
        
        Returns:
            HEF: A HEF (Hailo Executable File) containing the model.
        """
        return self.hef

    def get_input_shape(self) -> Tuple[int, ...]:
        """
        Get the shape of the model's input layer.

        Returns:
            Tuple[int, ...]: Shape of the model's input layer.
        """
        return self.hef.get_input_vstream_infos()[0].shape  # Assumes one input
        
    def run(self,metric = None) -> None:
        with self.infer_model.configure() as configured_infer_model:
            
            batch_data = self.input_queue.get()
            preprocessed_batch = batch_data

            bindings_list = []
            
            # if data is batched create bindings for every batch
            if self.is_data_batched:
                for frame in preprocessed_batch:
                    bindings = self._create_bindings(configured_infer_model)
                    bindings.input().set_buffer(np.array(frame))
                    bindings_list.append(bindings)
            else:
                bindings = self._create_bindings(configured_infer_model)
                bindings.input().set_buffer(np.array(preprocessed_batch))
                bindings_list.append(bindings)

            configured_infer_model.wait_for_async_ready(timeout_ms=10000)
            job = configured_infer_model.run_async(
                bindings_list, partial(
                    self.callback,
                    input_batch=preprocessed_batch,
                    bindings_list=bindings_list
                )
            )
            job.wait(10000)  # Wait for the last job

    def _get_output_type_str(self, output_info) -> str:
        if self.output_type is None:
            return str(output_info.format.type).split(".")[1].lower()
        else:
            self.output_type[output_info.name].lower()

    def _create_bindings(self, configured_infer_model) -> object:
        """
        Create bindings for input and output buffers.

        Args:
            configured_infer_model: The configured inference model.

        Returns:
            object: Bindings object with input and output buffers.
        """
        if self.output_type is None:
            output_buffers = {
                output_info.name: np.empty(
                    self.infer_model.output(output_info.name).shape,
                    dtype=(getattr(np, self._get_output_type_str(output_info)))
                )
            for output_info in self.hef.get_output_vstream_infos()
            }
        else:
            output_buffers = {
                name: np.empty(
                    self.infer_model.output(name).shape, 
                    dtype=(getattr(np, self.output_type[name].lower()))
                )
            for name in self.output_type
            }
        return configured_infer_model.create_bindings(
            output_buffers=output_buffers
        )
        
    def _set_input_type(self, input_type: Optional[str] = None) -> None:
        """
        Set the input type for the HEF model. If the model has multiple inputs,
        it will set the same type of all of them.

        Args:
            input_type (Optional[str]): Format type of the input stream.
        """
        self.infer_model.input().set_format_type(getattr(FormatType, input_type))
    
    def _set_output_type(self, output_type_dict: Optional[Dict[str, str]] = None) -> None:
        """
        Set the output type for the HEF model. If the model has multiple outputs,
        it will set the same type for all of them.

        Args:
            output_type_dict (Optional[dict[str, str]]): Format type of the output stream.
        """
        for output_name, output_type in output_type_dict.items():
            self.infer_model.output(output_name).set_format_type(
                getattr(FormatType, output_type)
            )