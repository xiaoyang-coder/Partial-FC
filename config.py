from easydict import EasyDict as edict

config = edict()
config.embedding_size = 512
config.bn_mom = 0.9
config.workspace = 256
config.net_se = 0
config.net_act = 'prelu'
config.net_unit = 3
config.net_input = 1
config.net_output = 'E'
config.frequent = 200
config.verbose = 2000
config.data_shape = (3, 112, 112)
config.loss_s = 64.0
config.loss_m1 = 1.0
config.loss_m2 = 0.0
config.loss_m3 = 0.40
config.lr_steps = '20000,28000'
config.sample_ratio = 1.0
config.fp16 = False
config.scheduler_type = 'sgd'
config.val_targets = ['lfw', 'cfp_fp', 'agedb_30']
config.rec_list = [
    # '/anxiang/datasets/webface/'
    # /root/xy/face/faces_emore
    # /root/face_datasets/webface/
]
config.momentum=0.9
config.weight_decay=5e-4
config.head_name = ['webface']
config.memory_lr_scale = 1.0
config.num_classes = 10572
# config.num_classes = 85742
config.batch_size = 64
config.max_update = 32000
config.lr =0.1
config.warmup_steps = config.max_update // 5
config.backbone_lr = 0.1
config.backbone_final_lr = config.backbone_lr / 1000
config.memory_bank_lr = 0.1
config.memory_bank_final_lr = config.backbone_lr / 1000



def get_sub_class(rank,world_size):
    config.num = config.num_classes//world_size + \
        int(rank < config.num_classes % world_size)
    config.s = config.num_classes//world_size*rank + \
        min(rank, config.num_classes % world_size)
    return config.s,config.num
