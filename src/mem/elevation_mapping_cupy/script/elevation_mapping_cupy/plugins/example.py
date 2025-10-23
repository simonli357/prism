import cupy as cp
from typing import List, Tuple
from elevation_mapping_cupy.plugins.plugin_manager import PluginBase
from elevation_mapping_cupy.plugins.custom.infer import InferenceHandler

class MultiOutputPlugin(PluginBase):
    def __init__(self, cell_n: int = 100,  
                 **kwargs):
        super().__init__()
        self.processed_elevation_buffer = cp.zeros((cell_n, cell_n), dtype=cp.float32)
        self.processed_rgb_buffer = cp.zeros((cell_n, cell_n), dtype=cp.float32) # Buffer for the packed float32 RGB
        # self.processed_rgb_traversability_buffer = cp.zeros((cell_n, cell_n), dtype=cp.float32) # Buffer for the RGB traversability
        self.input_elevation = None
        self.input_rgb_packed_float = None
        
        self.last_update_time = -1.0
        
        print("Initializing InferenceHandler...")
        model_path = "/media/slsecret/T7/carla3/runs/all357_cnn/checkpoint_best.pt"
        self.handler = InferenceHandler(model_path=model_path, device='cuda')
        print("MultiOutputPlugin with InferenceHandler Initialized.")

    def _get_layer_index(self, layer_names: List[str], target_name: str) -> int:
        try:
            return layer_names.index(target_name)
        except ValueError:
            return -1

    def _run_computation(self, elevation_map: cp.ndarray, semantic_map: cp.ndarray, rgb_idx: int):
        pred_rgb_packed, pred_elev, _ = self.handler.run_inference(
            self.input_rgb_packed_float, 
            self.input_elevation
        )
        
        self.processed_elevation_buffer[...] = pred_elev
        self.processed_rgb_buffer[...] = pred_rgb_packed
        # self.processed_rgb_traversability_buffer[...] = pred_rgb_trav
        
    def __call__(
        self,
        elevation_map: cp.ndarray,
        layer_names: List[str],
        plugin_layers: cp.ndarray,
        plugin_layer_names: List[str],
        semantic_map: cp.ndarray,
        semantic_layer_names: List[str],
        *args,
        **kwargs,
    ) -> cp.ndarray:
        
        layer_name = kwargs['layer_name']
        
        rgb_idx = self._get_layer_index(semantic_layer_names, "rgb")
        if rgb_idx == -1:
            print(f"Warning: Input layer 'rgb' not found. Returning empty buffer for '{layer_name}'.")
            return self.processed_elevation_buffer

        time_idx = self._get_layer_index(layer_names, "time")
        current_map_time = cp.max(elevation_map[time_idx]) if time_idx != -1 else 0

        self.input_elevation = elevation_map[0]
        self.input_rgb_packed_float = semantic_map[rgb_idx]
        
        # if layer_name == "elevation2":
        if True:
            self._run_computation(elevation_map, semantic_map, rgb_idx)

        if layer_name == "elevation2":
            return self.processed_elevation_buffer
        elif layer_name == "rgb2":
            return self.processed_rgb_buffer
        else:
            print(f"Warning: Requested unknown layer '{layer_name}'. Returning empty buffer.")
            return cp.zeros_like(elevation_map[0])