import graphviz

def generate_architecture_diagram():
    """
    Generates a polished, high-level block diagram for the 
    CNNCorrectionNetSingle using Graphviz.
    """
    
    # --- 1. Graph Initialization ---
    dot = graphviz.Digraph('CNNCorrectionNetSingle', comment='CNN Correction Net Architecture')
    
    # Global graph attributes
    dot.attr(
        rankdir='TB',        # Changed from 'LR' to 'TB' (Top-to-Bottom)
        splines='ortho',     # Orthogonal (right-angle) edges
        nodesep='0.6',       # Space between nodes on same rank
        ranksep='1.0',       # Space between ranks (can be smaller for TB)
        fontname='Helvetica' # Cleaner font
    )
    
    # Global node attributes
    dot.attr('node', 
        fontname='Helvetica',
        fontsize='16'  # Increased from 12
    )
    
    # Global edge attributes
    dot.attr('edge', 
        fontname='Helvetica',
        fontsize='14',  # Increased from 10
        penwidth='1.5'  # Thicker lines
    )

    # --- 2. Node Style Definitions ---
    style = {
        'input': {'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#a9d1f7'},
        'output': {'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#a9f7d1'},
        'block': {
            'shape': 'box', 
            'style': 'filled,rounded', 
            'fillcolor': '#f7e8a9', 
            'width': '2.5',  # Set a min width, height will be auto
            'height': '1.2'  # Set a min height
        },
        'op': {
            'shape': 'circle', 
            'style': 'filled', 
            'fillcolor': '#f7a9a9', 
            'width': '0.7',  # Make op nodes smaller
            'height': '0.7',
            'fontsize': '12' # Increased from 10
        },
        'mask': {'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#d9a9f7'}
    }

    # --- 3. Define Nodes ---
    
    # Input/Output
    dot.node('input', 'Input\n(sem, elev, mask)\n[B, C, H, W]', **style['input'])
    dot.node('output', 'Output\n(sem_logits, elev)\n[B, C_out, H, W]', **style['output'])

    # Main Blocks
    dot.node('stem', 'Stem\n(Conv + ResBlock)', **style['block'])
    dot.node('stage1', 'Stage 1\n(ResBlocks + ASPP)', **style['block'])
    dot.node('stage2', 'Stage 2\n(ResBlocks + ASPP)', **style['block'])
    dot.node('heads', 'Residual Heads\n(2x 1x1 Conv)', **style['block'])
    
    # Operation Nodes (the "glue")
    dot.node('slice_input', 'Split\nInput', **style['op'])
    dot.node('s1_head', '1x1 Conv\n(Edit Logits)', **style['mask'])
    dot.node('sigmoid', 'Sigmoid', **style['op'])
    dot.node('cat_s2', 'Concat', **style['op'])
    dot.node('gate_mul', 'Gate\n(Multiply)', **style['op'])
    dot.node('final_add', 'Add\nResiduals', **style['op'])

    
    # --- 4. Define Edges (Data Flow) ---
    
    # To manage the vertical layout better, we can use 'subgraphs'
    # to hint at the desired ranking (horizontal alignment)
    
    # Input -> Stem -> Stage 1
    dot.edge('input', 'stem', label=' x_last')
    dot.edge('stem', 'stage1', label=' f0')
    
    # Stage 1 -> Edit Mask
    dot.edge('stage1', 's1_head')
    dot.edge('s1_head', 'sigmoid', label=' edit_logits')

    # This subgraph helps align f0 and edit_prob horizontally
    with dot.subgraph(name='cluster_s2_input') as c:
        c.attr(rank='same') # Tell graphviz to put these on the same "rank"
        c.node('cat_s2')
        # We also need to re-route the f0 edge to cat_s2
        dot.edge('stem', 'cat_s2', label=' f0')
        dot.edge('sigmoid', 'cat_s2', label=' edit_prob')

    # Stage 2 path
    dot.edge('cat_s2', 'stage2', label=' f2_in')
    dot.edge('stage2', 'heads', label=' f2')
    
    # Gating path
    dot.edge('heads', 'gate_mul', label=' d_sem, d_elev')
    dot.edge('sigmoid', 'gate_mul', label=' edit_prob') # This edge now flows down
    
    # Final Identity Connection
    dot.edge('input', 'slice_input')
    
    # This subgraph helps align the two inputs to the final add
    with dot.subgraph(name='cluster_final_add') as c:
        c.attr(rank='same')
        c.node('gate_mul')
        c.node('slice_input')
        
    dot.edge('gate_mul', 'final_add', label=' gated_residuals')
    dot.edge('slice_input', 'final_add', label=' sem_in, elev_in')
    
    # Output
    dot.edge('final_add', 'output')

    
    # --- 5. Render Graph ---
    try:
        dot.render('cnn_architecture_vertical', format='pdf', cleanup=True, view=True)
        print("Successfully generated 'cnn_architecture_vertical.pdf'")
    except Exception as e:
        print(f"Error rendering graph. Is Graphviz installed? {e}")

if __name__ == "__main__":
    generate_architecture_diagram()

