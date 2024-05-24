voxel_size = .03 #NOTE: Increased voxel size
n_points = 262144

model = dict(
    type='MinkSingleStage3DDetector',
    voxel_size=voxel_size,
    backbone=dict(type='MinkResNet', in_channels=3, depth=34, max_channels=128, norm='batch'),
    neck=dict(
        type='TR3DNeck',
        in_channels=(64, 128, 128, 128),
        out_channels=128),
    head=dict(
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=12,
        n_classes=1,
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level=[1]),
        bbox_loss=dict(type='ChamferDistance')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=.3))

optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
# optimizer = dict(type='SGD', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

dataset_type = 'PhysionRandomFrameDataset'
data_root = '/media/kalyanav/Venkat/support_data/'
# class_names = ['cloth_square', 'buddah', 'bowl', 'cone', 'cube', 'cylinder', 'dumbbell', 'octahedron', 'pentagon', 'pipe', 'platonic', 'pyramid', 'sphere', 'torus', 'triangular_prism']
class_names = ['object']
train_pipeline = [
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='DEPTH',
    #     shift_height=False,
    #     use_color=True,
    #     load_dim=6,
    #     use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='PointSample', num_points=n_points),
    # dict(
    #     type='RandomFlip3DPhysion'
    # ),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=False,
    #     flip_ratio_bev_horizontal=.5,
    #     flip_ratio_bev_vertical=.0),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-.523599, .523599],
    #     scale_ratio_range=[.85, 1.15],
    #     translation_std=[.1, .1, .1],
    #     shift_height=False),
    # dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='DEPTH',
    #     shift_height=False,
    #     use_color=True,
    #     load_dim=6,
    #     use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(512, 512),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample', num_points=n_points),
            # dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    persistent_workers=False,
    num_frames_per_file = 10,
    train=
        dict(
            type=dataset_type,
            modality=dict(use_camera=False, use_lidar=True),
            data_root=data_root,
            ann_file=data_root+'train_support_may.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Physion'),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'val_support_may.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Physion'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'val_support_may.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Physion'))
