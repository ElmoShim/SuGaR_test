from sugar_trainers.refine import refined_training

class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self


if __name__ == "__main__":
    # ----- Refine SuGaR -----
    refined_args = AttrDict({
        'scene_path': "input/A11",
        'checkpoint_path': "output/vanilla_gs/A11/",
        'mesh_path': "/mnt/DAS2/proj_3dgs/sugar/baseline.ply",      
        'output_dir': "/mnt/DAS2/proj_3dgs/sugar/output",
        'normal_consistency_factor': 0.0,    
        'gaussians_per_triangle': 6,        
        'n_vertices_in_fg': 1000000,
        'refinement_iterations': 15000,
        'bboxmin': None,
        'bboxmax': None,
        'export_ply': False,
        'eval': False,
        'gpu': 0,
        'white_background': False,
    })
    refined_sugar_path = refined_training(refined_args)

    