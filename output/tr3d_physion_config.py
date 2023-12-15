voxel_size = 0.01
n_points = 65536
model = dict(
    type='MinkSingleStage3DDetector',
    voxel_size=0.01,
    backbone=dict(
        type='MinkResNet',
        in_channels=3,
        depth=34,
        max_channels=128,
        norm='batch'),
    neck=dict(
        type='TR3DNeck', in_channels=(64, 128, 128, 128), out_channels=128),
    head=dict(
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=8,
        n_classes=15,
        voxel_size=0.01,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level=[0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0]),
        bbox_loss=dict(type='RotatedIoU3DLoss', mode='diou',
                       reduction='none')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=0.5, score_thr=0.01))
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'output'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'SUNRGBDDataset'
data_root = '/home/kalyanav/MS_thesis/tr3d_data/physion/'
class_names = [
    'cloth_square', 'buddah', 'bowl', 'cone', 'cube', 'cylinder', 'dumbbell',
    'octahedron', 'pentagon', 'pipe', 'platonic', 'pyramid', 'sphere', 'torus',
    'triangular_prism'
]
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='PointSample', num_points=65536),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.0),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'cloth_square', 'buddah', 'bowl', 'cone', 'cube', 'cylinder',
            'dumbbell', 'octahedron', 'pentagon', 'pipe', 'platonic',
            'pyramid', 'sphere', 'torus', 'triangular_prism'
        ]),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample', num_points=65536),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'cloth_square', 'buddah', 'bowl', 'cone', 'cube',
                    'cylinder', 'dumbbell', 'octahedron', 'pentagon', 'pipe',
                    'platonic', 'pyramid', 'sphere', 'torus',
                    'triangular_prism'
                ],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type='SUNRGBDDataset',
            modality=dict(use_camera=False, use_lidar=True),
            data_root='/home/kalyanav/MS_thesis/tr3d_data/physion/',
            ann_file='/home/kalyanav/MS_thesis/tr3d_data/physion/train.pkl',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='DEPTH',
                    shift_height=False,
                    use_color=True,
                    load_dim=6,
                    use_dim=[0, 1, 2, 3, 4, 5]),
                dict(type='LoadAnnotations3D'),
                dict(type='PointSample', num_points=65536),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.0),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.523599, 0.523599],
                    scale_ratio_range=[0.85, 1.15],
                    translation_std=[0.1, 0.1, 0.1],
                    shift_height=False),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'cloth_square', 'buddah', 'bowl', 'cone', 'cube',
                        'cylinder', 'dumbbell', 'octahedron', 'pentagon',
                        'pipe', 'platonic', 'pyramid', 'sphere', 'torus',
                        'triangular_prism'
                    ]),
                dict(
                    type='Collect3D',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            filter_empty_gt=False,
            classes=[
                'cloth_square', 'buddah', 'bowl', 'cone', 'cube', 'cylinder',
                'dumbbell', 'octahedron', 'pentagon', 'pipe', 'platonic',
                'pyramid', 'sphere', 'torus', 'triangular_prism'
            ],
            box_type_3d='Depth')),
    val=dict(
        type='SUNRGBDDataset',
        modality=dict(use_camera=False, use_lidar=True),
        data_root='/home/kalyanav/MS_thesis/tr3d_data/physion/',
        ann_file='/home/kalyanav/MS_thesis/tr3d_data/physion/val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='PointSample', num_points=65536),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'cloth_square', 'buddah', 'bowl', 'cone', 'cube',
                            'cylinder', 'dumbbell', 'octahedron', 'pentagon',
                            'pipe', 'platonic', 'pyramid', 'sphere', 'torus',
                            'triangular_prism'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=[
            'cloth_square', 'buddah', 'bowl', 'cone', 'cube', 'cylinder',
            'dumbbell', 'octahedron', 'pentagon', 'pipe', 'platonic',
            'pyramid', 'sphere', 'torus', 'triangular_prism'
        ],
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type='SUNRGBDDataset',
        modality=dict(use_camera=False, use_lidar=True),
        data_root='/home/kalyanav/MS_thesis/tr3d_data/physion/',
        ann_file='/home/kalyanav/MS_thesis/tr3d_data/physion/val.pkl',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                use_color=True,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(type='PointSample', num_points=65536),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'cloth_square', 'buddah', 'bowl', 'cone', 'cube',
                            'cylinder', 'dumbbell', 'octahedron', 'pentagon',
                            'pipe', 'platonic', 'pyramid', 'sphere', 'torus',
                            'triangular_prism'
                        ],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=[
            'cloth_square', 'buddah', 'bowl', 'cone', 'cube', 'cylinder',
            'dumbbell', 'octahedron', 'pentagon', 'pipe', 'platonic',
            'pyramid', 'sphere', 'torus', 'triangular_prism'
        ],
        test_mode=True,
        box_type_3d='Depth'))
gpu_ids = [0]
