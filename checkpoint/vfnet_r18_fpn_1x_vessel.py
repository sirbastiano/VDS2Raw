dataset_type = 'CocoDataset'
data_root = 'data/vessels/'
metainfo = dict(classes=('vessel', ), palette=[(220, 20, 60)])
backend_args = None
IMG_SCALE = (2048, 2048)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        color_type='color',
        imdecode_backend='tifffile',
        backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        color_type='color',
        imdecode_backend='tifffile',
        backend_args=None),
    dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor','flip','height','width'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        data_root='data/vessels/',
        metainfo=dict(classes=('vessel', ), palette=[(220, 20, 60)]),
        ann_file='annotations/train.json',
        data_prefix=dict(img='imgs/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                to_float32=True,
                color_type='color',
                imdecode_backend='tifffile',
                backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(type='PackDetInputs')
        ],
        backend_args=None))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/vessels/',
        metainfo=dict(classes=('vessel', ), palette=[(220, 20, 60)]),
        ann_file='annotations/val.json',
        data_prefix=dict(img='imgs/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                to_float32=True,
                color_type='color',
                imdecode_backend='tifffile',
                backend_args=None),
            dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ],
        backend_args=None))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root='data/vessels/',
        metainfo=dict(classes=('vessel', ), palette=[(220, 20, 60)]),
        ann_file='annotations/test.json',
        data_prefix=dict(img='imgs/'),
        test_mode=True,
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                to_float32=True,
                color_type='color',
                imdecode_backend='tifffile',
                backend_args=None),
            dict(type='Resize', scale=(2048, 2048), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img', 'img_id', 'img_path', 'img_shape', 'ori_shape', 'scale', 'scale_factor', 'keep_ratio', 'homography_matrix', 'gt_bboxes', 'gt_ignore_flags', 'gt_bboxes_labels'))
        ],
        backend_args=None))
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/vessels/annotations/val.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    format_only=False,
    ann_file='data/vessels/annotations/test.json',
    outfile_prefix='./work_dirs/vessel_detection/test')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=240, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=120,
        by_epoch=True,
        milestones=[80, 130, 160],
        gamma=0.7)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001))
auto_scale_lr = dict(enable=False, base_batch_size=2)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='DetVisualizationHook',
        draw=True,
        test_out_dir=
        '/home/roberto/PythonProjects/S2RAWVessel/checkpoints/vfnet_r18_fpn_1x_vessel/20230518_152156_0.0005/inference'
    ))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(
            type='WandbVisBackend',
            init_kwargs=dict(
                project='S2RAWVessel_TestMode',
                group='vfnet_r18_fpn_1x_vessel_TEST'))
    ],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = '/home/roberto/PythonProjects/S2RAWVessel/checkpoints/vfnet_r18_fpn_1x_vessel/20230518_152156_0.0005/epoch_239.pth'
resume = False
model = dict(
    type='VFNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[200, 154, 116],
        std=[22, 24, 27],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='VFNetHead',
        num_classes=1,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        center_sampling=False,
        dcn_on_last_conv=False,
        use_atss=True,
        use_vfl=True,
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            iou_weighted=True,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5),
        loss_bbox_refine=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
launcher = 'none'
work_dir = '/home/roberto/PythonProjects/S2RAWVessel/checkpoints/vfnet_r18_fpn_1x_vessel/20230518_152156_0.0005'
