import cupy as cp
from typing import List, Tuple
from elevation_mapping_cupy.plugins.plugin_manager import PluginBase
from elevation_mapping_cupy.plugins.custom.infer import InferenceHandler

class MultiOutputPlugin(PluginBase):
    def __init__(self, 
                 cell_n: int = 100,  
                 rgb_weight: float = 1.0,
                 geom_weight: float = 0.0,
                 **kwargs):
        super().__init__()
        
        self.input_elevation = None
        self.input_rgb_packed_float = None
        
        print("Initializing InferenceHandler...")
        model_path = "/media/slsecret/T7/carla3/runs/all357_cnn/checkpoint_best.pt"
        # model_path = "/media/slsecret/T7/carla3/runs/all357_unet/checkpoint_best.pt"
        
        self.handler = InferenceHandler(
            model_path=model_path, 
            device='cuda', 
            cell_n=cell_n,
            rgb_weight=rgb_weight,
            geom_weight=geom_weight
        )
        print("MultiOutputPlugin with InferenceHandler Initialized.")

    def _get_layer_index(self, layer_names: List[str], target_name: str) -> int:
        try:
            return layer_names.index(target_name)
        except ValueError:
            return -1

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
            return cp.zeros_like(elevation_map[0])

        time_idx = self._get_layer_index(layer_names, "time")
        current_map_time = cp.max(elevation_map[time_idx]) if time_idx != -1 else 0

        self.input_elevation = elevation_map[0]
        self.input_rgb_packed_float = semantic_map[rgb_idx]
        
        self.handler.run_inference(
            current_map_time, 
            self.input_rgb_packed_float, 
            self.input_elevation
        )

        # min_val_buf = cp.min(self.handler.processed_elevation_buffer)
        # max_val_buf = cp.max(self.handler.processed_elevation_buffer)
        # avg_val_buf = cp.mean(self.handler.processed_elevation_buffer)
        # median_val_buf = cp.median(self.handler.processed_elevation_buffer)
        # print(f"[{layer_name}] Processed Elevation Buffer: min={min_val_buf:.4f}, max={max_val_buf:.4f}, avg={avg_val_buf:.4f}, median={median_val_buf:.4f}")

        if layer_name == "elevation2":
            return self.handler.processed_elevation_buffer
            
        elif layer_name == "rgb2":
            return self.handler.processed_rgb_buffer
        elif layer_name == "traversability2":
            return self.handler.processed_geom_traversability_buffer
        elif layer_name == "rgb_traversability":
            return self.handler.processed_rgb_traversability_buffer
        elif layer_name == "combined_cost":
            return self.handler.processed_combined_cost_buffer
        else:
            print(f"Warning: Requested unknown layer '{layer_name}'. Returning empty buffer.")
            return cp.zeros_like(elevation_map[0])