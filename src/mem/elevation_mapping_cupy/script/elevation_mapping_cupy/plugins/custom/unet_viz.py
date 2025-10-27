import graphviz

def generate_unet_lstm_diagram():
    """
    Generates a polished, high-level block diagram for the 
    UNetConvLSTM using Graphviz.
    """
    
    # --- 1. Graph Initialization ---
    dot = graphviz.Digraph('UNetConvLSTM', comment='UNet ConvLSTM Architecture')
    
    # Global graph attributes
    dot.attr(
        rankdir='TB',        # Top-to-Bottom layout
        splines='ortho',     # Orthogonal (right-angle) edges
        nodesep='0.6',       # Space between nodes on same rank
        ranksep='1.2',       # Space between ranks
        fontname='Helvetica' # Cleaner font
    )
    
    # Global node attributes
    dot.attr('node', 
        fontname='Helvetica',
        fontsize='16'
    )
    
    # Global edge attributes
    dot.attr('edge', 
        fontname='Helvetica',
        fontsize='14',
        penwidth='1.5'
    )

    # --- 2. Node Style Definitions ---
    style = {
        'input': {'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#a9d1f7'},
        'output': {'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#a9f7d1'},
        'block': {
            'shape': 'box', 
            'style': 'filled,rounded', 
            'fillcolor': '#f7e8a9', 
            'width': '3.0',
            'height': '1.2'
        },
        'op': {
            'shape': 'circle', 
            'style': 'filled', 
            'fillcolor': '#f7a9a9', 
            'width': '0.9',
            'height': '0.9',
            'fontsize': '12'
        },
    }

    # --- 3. Define Nodes ---
    
    # Input/Output
    dot.node('input', 'Input\n[B, T, C, H, W]', **style['input'])
    dot.node('output', 'Output\n(sem_logits, elev)\n[B, C_out, H, W]', **style['output'])

    # Encoder Blocks (Looped)
    dot.node('enc1', 'Encoder 1 (t)\n(ConvBlock)\n[B, base, H, W]', **style['block'])
    dot.node('enc2', 'Encoder 2 (t)\n(Down)\n[B, 2base, H/2, W/2]', **style['block'])
    dot.node('enc3', 'Encoder 3 (t)\n(Down)\n[B, 4base, H/4, W/4]', **style['block'])
    
    # Bottleneck
    dot.node('stack', 'Stack (T frames)', **style['op'])
    dot.node('lstm', 'ConvLSTM\n(Bottleneck)', **style['block'])

    # Decoder Blocks
    dot.node('up2', 'Decoder 2\n(Up)\n[B, 2base, H/2, W/2]', **style['block'])
    dot.node('up1', 'Decoder 1\n(Up)\n[B, base, H, W]', **style['block'])
    
    # Heads
    dot.node('sem_head', 'Semantic Head\n(ConvBlock)', **style['block'])
    dot.node('elev_head', 'Elevation Head\n(ConvBlock)', **style['block'])
    
    # Head Operations
    dot.node('interp_d2', 'Interpolate\n(Bilinear)', **style['op'])
    dot.node('cat_elev', 'Concat (Elev)', **style['op'])
    dot.node('cat_out', 'Concat (Output)', **style['op'])
    

    # --- 4. Define Edges (Data Flow) ---
    
    # Encoder Path (conceptually looped)
    dot.edge('input', 'enc1', label=' x_t (looped T times)')
    dot.edge('enc1', 'enc2')
    dot.edge('enc2', 'enc3')
    
    # Path to LSTM
    dot.edge('enc3', 'stack', label=' f3 (all T frames)')
    dot.edge('stack', 'lstm', label=' feats [B,T,4b,H/4,W/4]')
    
    # Decoder Path (U-Net Skips)
    dot.edge('lstm', 'up2', label=' bottleneck [B,4b,H/4,W/4]')
    dot.edge('enc2', 'up2', label=' skip2 (last frame)')
    
    dot.edge('up2', 'up1', label=' d2 [B,2b,H/2,W/2]')
    dot.edge('enc1', 'up1', label=' skip1 (last frame)')

    # Semantic Head
    dot.edge('up1', 'sem_head', label=' d1 [B,base,H,W]')
    dot.edge('sem_head', 'cat_out', label=' sem_logits')

    # Elevation Head (Special Path)
    dot.edge('up1', 'cat_elev', label=' d1')
    dot.edge('up2', 'interp_d2', label=' d2')
    dot.edge('interp_d2', 'cat_elev', label=' d2_up')
    dot.edge('cat_elev', 'elev_head', label=' elev_in')
    dot.edge('elev_head', 'cat_out', label=' elev_pred')
    
    # Output
    dot.edge('cat_out', 'output')
    
    
    # --- 5. Add Ranks for U-Net Shape ---
    # This aligns encoder/decoder stages horizontally
    with dot.subgraph(name='rank_enc1_up1') as c:
        c.attr(rank='same')
        c.node('enc1')
        c.node('up1')
        
    with dot.subgraph(name='rank_enc2_up2') as c:
        c.attr(rank='same')
        c.node('enc2')
        c.node('up2')

    with dot.subgraph(name='rank_enc3_lstm') as c:
        c.attr(rank='same')
        c.node('enc3')
        c.node('stack')
        c.node('lstm')
        
    # --- 6. Render Graph ---
    try:
        dot.render('unet_lstm_architecture', format='pdf', cleanup=True, view=True)
        print("Successfully generated 'unet_lstm_architecture.pdf'")
    except Exception as e:
        print(f"Error rendering graph. Is Graphviz installed? {e}")

if __name__ == "__main__":
    generate_unet_lstm_diagram()
