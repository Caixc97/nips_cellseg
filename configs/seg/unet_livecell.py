norm_cfg = dict(type='BN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='ASPPHead',
        in_channels=64,
        in_index=4,
        channels=16,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
        ]),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict())

dataset_type = 'GTBBoxDataset'
data_root = '../data/LIVECell_dataset_2021/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

input_size = (128, 128)
train_pipeline = [
    dict(type='BoxJitter', prob=0.5),
    dict(type='ROIAlign', output_size=input_size),
    # dict(type='FlipRotate'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=input_size,
        flip=False,
        transforms=[
            dict(type='ROIAlign', output_size=input_size),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='BBoxFormat'),
            dict(type='Collect', keys=['img', 'bbox'])
        ]
    )
]
helper_dataset = dict(
    type='CellDataset',
    classes=(
        'shsy5y', 'a172', 'bt474', 'bv2', 'huh7', 'mcf7', 'skov3', 'skbr3'
    ),
    ann_file=data_root + 'val_8class.json',
    img_prefix=data_root + 'images/livecell_train_val_images',
    pipeline=[],
)
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=2,
    train=dict(
        type='GTBBoxDataset',
        img_dir=data_root + 'images/livecell_train_val_images',
        ann_file=data_root + 'train_8class.json',
        helper_dataset=helper_dataset,
        pipeline=train_pipeline
    ),
    val=dict(
        type='PredBBoxDataset',
        # score_thr=0.3,
        pred_file='work_dirs/yolox_x_livecell/val_preds.pkl',
        img_dir=data_root + 'images/livecell_train_val_images',
        ann_file=data_root + 'val_8class.json',
        helper_dataset=helper_dataset,
        pipeline=test_pipeline
    ),
    test=dict(
        type='PredBBoxDataset',
        pred_file='work_dirs/yolox_x_livecell/val_preds.pkl',
        img_dir=data_root + 'images/livecell_train_val_images',
        ann_file=data_root + 'val_8class.json',
        helper_dataset=helper_dataset,
        pipeline=test_pipeline
    )
)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=True)]
)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/unet/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive/deeplabv3_unet_s5-d16_ce-1.0-dice-3.0_64x64_40k_drive_20211210_201825-6bf0efd7.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05 / 16,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0)
        )
    )
)
optimizer_config = dict()
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5
)
runner = dict(type='EpochBasedRunner', max_epochs=1)
checkpoint_config = dict(interval=1, save_optimizer=False)
evaluation = dict(interval=1, metric='dummy', pre_eval=True)
