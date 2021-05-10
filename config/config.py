class Config(object):
    env = 'default'
    backbone = 'my_curtin_data'
    classify = 'softmax'
    num_classes = 340 # 52 # 659 # 8277 # 710 # 509 # 340
    metric = 'arc_margin'
    easy_margin = False
    use_se = True
    loss = 'cross_entry'

    display = True
    finetune = False

    train_root = 'D:\\data\\project\\java\\Kinect\\Lock3d\\Lock3d\\' # 'E:\\paperData\\' #  'E:\\paperData\\' #
    train_list = 'D:\\data\\project\\java\\Kinect\\Lock3d\\Lock3d\\rec_train_340.txt' # 'E:\\paperData\\Lock3d\\lock3d_train.txt'  # 'E:\\paperData\\curtinTrain.txt' #'E:\\paperData\\BFtrain.txt' #

    test_root = 'E:\\paperData\\'
    gallery_c = 'E:\\paperData\\curtinFace\\curtinGallery.txt' # 'D:\\data\\project\\java\\Kinect\\Lock3d\\Lock3d\\gallery.txt'
    swjtu_root = "E:\\paperData\\swjtu\\"
    probe_swjtu = 'E:\\paperData\\swjtu\\probe.txt'
    gallery_swjtu = 'E:\\paperData\\swjtu\\gallery.txt'
    c_probe_IE = "E:\\paperData\\curtinFace\\curtinProbeIE.txt"
    c_probe_PE = "E:\\paperData\\curtinFace\\curtinProbePE.txt"
    c_probe_OC = "E:\\paperData\\curtinFace\\curtinProbeOC.txt"
    probe_fe = 'D:\\data\\project\\java\\Kinect\\Lock3d\\Lock3d\\probe_fe.txt'
    probe_ps = 'D:\\data\\project\\java\\Kinect\\Lock3d\\Lock3d\\probe_ps.txt'
    probe_oc = 'D:\\data\\project\\java\\Kinect\\Lock3d\\Lock3d\\probe_oc.txt'
    probe_tm = 'D:\\data\\project\\java\\Kinect\\Lock3d\\Lock3d\\probe_tm.txt'
    probe_nu = 'D:\\data\\project\\java\\Kinect\\Lock3d\\Lock3d\\probe_nu.txt'

    test_list_fe = "D:\\data\\project\\java\\Kinect\\Lock3d\\lock3d_test_FE.txt"
    test_list_oc = "D:\\data\\project\\java\\Kinect\\Lock3d\\lock3d_test_OC.txt"
    test_list_ps = "D:\\data\\project\\java\\Kinect\\Lock3d\\lock3d_test_PS.txt"
    test_list_tm = "D:\\data\\project\\java\\Kinect\\Lock3d\\lock3d_test_TM.txt"

    checkpoints_path = './output/'
    load_model_path = ''

    save_interval = 10

    train_batch_size = 32  # batch size
    test_batch_size = 32

    input_shape = (3, 128, 128)

    optimizer = 'sgd' # 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0'
    num_workers = 4  # how many workers for loading data
    print_freq = 50  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 30
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
