"""
@Fire
https://github.com/fire717
"""

cfg = {
    ##### Global Setting
    'GPU_ID': '0,1',
    "num_workers": 4,
    "random_seed": 42,
    "cfg_verbose": True,
    "save_dir": '',
    "num_classes": 13,
    "width_mult": 1.0,
    "img_size": 192,
    'label': '',

    ##### Train Setting
    'pre-separated_data': True,
    'training_data_split': 80,
    "dataset": '',
    'balance_data': False,
    'log_interval': 10,
    'save_best_only': False,

    'pin_memory': True,
    'newest_ckpt': '',
    'th': 50,  # percentage of headsize
    'from_scratch': True,

    ##### Train Hyperparameters
    'learning_rate': 0.001,  # 1.25e-4
    'batch_size': 64,
    'epochs': 300,
    'optimizer': 'Adam',  # Adam  SGD
    'scheduler': 'MultiStepLR-70,100-0.1',  # default  SGDR-5-2  CVPR   step-4-0.8 MultiStepLR
    # multistepLR-<<>milestones>-<<decay multiplier>>
    'weight_decay': 0.001,  # 5.e-4,  # 0.0001,
    'class_weight': None,  # [1., 1., 1., 1., 1., 1., 1., ]
    'clip_gradient': 5,  # 1,
    'w_heatmap': 1,
    'w_bone': 20,
    'w_center': 1,
    'w_reg': 3,
    'w_offset': 1,

    ##### File paths
    'predict_output_path': '',
    'results_path': '',
    "img_path": '',
    "test_img_path": '',
    "eval_img_path": '',
    "eval_outputs": '',
    "eval_label_path": '',
    'train_label_path': '',
    'val_label_path': '',
    'ckpt': ''
}

