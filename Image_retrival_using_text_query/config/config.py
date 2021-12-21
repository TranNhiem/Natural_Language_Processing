from config.absl_mock import Mock_Flag


def read_cfg(mod="non_contrastive"):
    flags = Mock_Flag()
    base_cfg()
    wandb_set()


def base_cfg():
    flags = Mock_Flag()

    flags.DEFINE_integer(
        'IMG_HEIGHT', 224,
        'image height.')

    flags.DEFINE_integer(
        'IMG_WIDTH', 224,
        'image width.')

    flags.DEFINE_integer(
        'SEED', 26,  # 40, 26
        'random seed use for shuffle data Generate two same image ds_one & ds_two')

    flags.DEFINE_integer(
        'training_samples', 40.000,
        'size of training dataset MS-COCO 80K')

    flags.DEFINE_integer(
        'val_samples', 5000,
        'size of validation dataset MS-COCO val or sperate from Train data.')

    flags.DEFINE_integer(
        'training_samples', 40.000,
        'size of training dataset MS-COCO 80K')

    flags.DEFINE_integer(
        'images_per_file', 2000,
        'Splitting train folder each subset train folders.')

    flags.DEFINE_float(
        'num_captions', 2,
        'Number of captions will make training sample increasing X times')

    flags.DEFINE_integer(
        'train_batch_size', 200,
        'Train batch_size .')

    flags.DEFINE_integer(
        'val_batch_size', 200,
        'Validaion_Batch_size.')

    flags.DEFINE_integer(
        'train_epochs', 100,
        'Number of epochs to train for.')

    flags.DEFINE_string(
        'root_dir', "",
        'path_store_data MS-COCO dataset.')

    # flags.DEFINE_string(
    #     'train_path', '/data1/share/1K_New/train/',
    #     'Train dataset path.')

    # flags.DEFINE_string(
    #     'val_path', "/data1/share/1K_New/val/",
    #     'Validaion dataset path.')


def wandb_set():
    flags = Mock_Flag()
    flags.DEFINE_string(
        "wandb_project_name", "NLP_finetune_BERT_Architecture",
        "set the project name for wandb."
    )
    flags.DEFINE_string(
        "wandb_run_name", "Image_retrival_from_Text_Query",
        "set the run name for wandb."
    )
    flags.DEFINE_enum(
        'wandb_mod', 'run', ['run', 'dryrun'],
        'update the to the wandb server or not')
