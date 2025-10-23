all the inferences results are in /media/slsecret/T7/carla3/runs$ ls
all357_cnn  all357_deeplabv3p  all357_unet  all357_unet_attention

each contains all the checkpoints, the manifests in json, and the inference results in the folder inference_on_test_set
/media/slsecret/T7/carla3/runs/all357_unet$ ls
checkpoint_best.pt       checkpoint_epoch_012.pt  checkpoint_epoch_024.pt
checkpoint_epoch_001.pt  checkpoint_epoch_013.pt  checkpoint_epoch_025.pt
checkpoint_epoch_002.pt  checkpoint_epoch_014.pt  checkpoint_epoch_026.pt
checkpoint_epoch_003.pt  checkpoint_epoch_015.pt  checkpoint_epoch_027.pt
checkpoint_epoch_004.pt  checkpoint_epoch_016.pt  checkpoint_epoch_028.pt
checkpoint_epoch_005.pt  checkpoint_epoch_017.pt  checkpoint_epoch_029.pt
checkpoint_epoch_006.pt  checkpoint_epoch_018.pt  checkpoint_epoch_030.pt
checkpoint_epoch_007.pt  checkpoint_epoch_019.pt  class_weights.json
checkpoint_epoch_008.pt  checkpoint_epoch_020.pt  inference_on_test_set
checkpoint_epoch_009.pt  checkpoint_epoch_021.pt  manifest.config.json
checkpoint_epoch_010.pt  checkpoint_epoch_022.pt  manifest.training.

the inference_on_test_set folder contains a shard, in which the files are organized like this
run1_000000300.gt_color14.png     run1_000000301.pred_elev.npy
run1_000000300.gt_elev_viz.png    run1_000000301.pred_elev_viz.png
run1_000000300.pred_color14.png   run1_000000302.gt_color14.png
run1_000000300.pred_elev.npy      run1_000000302.gt_elev_viz.png
run1_000000300.pred_elev_viz.png  run1_000000302.pred_color14.png
run1_000000301.gt_color14.png     run1_000000302.pred_elev.npy
run1_000000301.gt_elev_viz.png    run1_000000302.pred_elev_viz.png
run1_000000301.pred_color14.png

where .npy files are the elevation data in numpy format, _viz.png are the visualization of the elevation data as heatmap images, and _color14.png are the semantic segmentation results with 14 classes in color format. The gt files are the ground truth data, and the pred files are the predicted data from the model.

now in the ground truth testset folder we have similar shards: /media/slsecret/T7/carla3/data_split357/test$ ls
town0_gridmap_wds_remapped_shard-000004.tar
town1_gridmap_wds_remapped_shard-000002.tar
town1_gridmap_wds_remapped_shard-000005.tar
town1_gridmap_wds_remapped_shard-000007.tar
town2_gridmap_wds_remapped_shard-000001.tar
town3_gridmap_wds_remapped_shard-000012.tar
town4_gridmap_wds_remapped_shard-000012.tar
town5_gridmap_wds_remapped_shard-000002.tar
town7_gridmap_wds_remapped_shard-000005.tar

in which each shard contains data like this
run1_000001200.gt_elev.npy         run1_000001201.gt_elev.npy
run1_000001200.gt_elev_viz.png     run1_000001201.gt_elev_viz.png
run1_000001200.gt_onehot14.npy     run1_000001201.gt_onehot14.npy
run1_000001200.gt_rgb.png          run1_000001201.gt_rgb.png
run1_000001200.meta.json           run1_000001201.meta.json
run1_000001200.tr_elev.npy         run1_000001201.tr_elev.npy
run1_000001200.tr_inpaint.npy      run1_000001201.tr_inpaint.npy
run1_000001200.tr_inpaint_viz.png  run1_000001201.tr_inpaint_viz.png
run1_000001200.tr_onehot14.npy     run1_000001201.tr_onehot14.npy
run1_000001200.tr_rgb.png          run1_000001201.tr_rgb.png

where tr_elev.npy is the input training data from the 'elevation' layer and tr_inpaint.npy is the input training data from the 'inpainted' layer. gt_elev.npy is the ground truth elevation data. gt_onehot14.npy is the ground truth semantic segmentation in onehot format with 14 classes, tr_onehot14.npy is the input training semantic segmentation in onehot format with 14 classes.

