from easydict import EasyDict as edict





H36M_CONF = edict({
    'train': {
        'sid': [1, 5, 6, 8],
        'sampling': 5
    },
    'test': {
        'sid': [9, 11],
        'sampling': 64
    },
    'val': {
        'sid': [7],
        'sampling': 64
    },
    'action': {
        'names': ['Directions', 'Discussion', 'Eating',
                  'Greeting', 'Phoning', 'Posing', 'Purchases',
                  'Sitting', 'SittingDown', 'Smoking', 'Photo',
                  'Waiting', 'Walking', 'WalkDog', 'WalkTogether'],
        'ids': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    },
    'joints': {
        'number': 18,
        'root_idx': 0,
        'l_shoulder_idx': 11,
        'r_shoulder_idx': 14,
        'interest': 14
    },
    'openpose':{
        'left_hip': 8,
        'right_hip': 11
    },
    'bbox_3d': (2000, 2000, 2000),
    'max_size': 1000,
    'input_size': 368,
    'output_size': 46,
    'depth_dim': 46
})
