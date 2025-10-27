import cupy as cp
from typing import List, Tuple
from elevation_mapping_cupy.plugins.plugin_manager import PluginBase
from .custom.color_mapping import color28_to_onehot14, onehot14_to_color, SEM_CHANNELS, onehot14_to_traversability

class CombinedPlugin(PluginBase):
    def __init__(self, 
                 cell_n: int = 300,
                 rgb_weight: float = 1.0,
                 geom_weight: float = 0.0,
                 **kwargs):
        super().__init__()
        
        self.output_buffer = cp.zeros((cell_n, cell_n), dtype=cp.float32)
        
        self.rgb_weight = float(rgb_weight)
        self.geom_weight = float(geom_weight)
        
        print(f"CombinedPlugin Initialized with rgb_weight: {self.rgb_weight}, geom_weight: {self.geom_weight}")

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
        
        geom_trav_layer = self.get_layer_data(
            elevation_map, layer_names, 
            plugin_layers, plugin_layer_names,
            semantic_map, semantic_layer_names,
            "traversability2"  
        )
        
        rgb_semantic_layer = self.get_layer_data(
            elevation_map, layer_names, 
            plugin_layers, plugin_layer_names,
            semantic_map, semantic_layer_names,
            "rgb2"  
        )

        if geom_trav_layer is None:
            print("Warning: CombinedPlugin could not find 'traversability2' input layer.")
            return self.output_buffer 
            
        if rgb_semantic_layer is None:
            print("Warning: CombinedPlugin could not find 'rgb2' input layer.")
            return self.output_buffer 

        rgb_trav_layer = onehot14_to_traversability(rgb_semantic_layer)

        cp.multiply(geom_trav_layer, self.geom_weight, out=self.output_buffer)

        self.output_buffer += (rgb_trav_layer * self.rgb_weight)
        
        return self.output_buffer