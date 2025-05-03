from .data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_PEMS, \
    Dataset_Pred, Dataset_M4, Dataset_Custom_frets, Dataset_Covid, Dataset_Custom2
from .dataset_physio import real_Physio_Dataset
from .dataset_pm25 import PM25_Dataset

from torch.utils.data import DataLoader
import copy

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,
    'PEMS': Dataset_PEMS,
    'custom': Dataset_Custom,
    'custom2': Dataset_Custom2,
    # 'air': PM25_Dataset,
    'air': Dataset_Custom,
    # 'physio': real_Physio_Dataset,
    'physio': Dataset_Custom,
    'm4': Dataset_M4,
    'covid': Dataset_Covid,
    'ECG': Dataset_Custom_frets,
    'metr': Dataset_Custom_frets,
}


def data_provider(args, flag, test_batch_size=None):
    assert flag in ['train', 'valid', 'test', 'pred']

    if args.task_name == 'imputation':
        # change args
        args2 = copy.deepcopy(args)
        args2.pred_len = 1
    else:
        args2 = args

    Data = data_dict[args2.data]
    timeenc = 0 if args2.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # do not drop
        batch_size = test_batch_size if test_batch_size else args2.test_batch_size
        # bsz=1 for evaluation
        freq = args2.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = test_batch_size if test_batch_size else args2.test_batch_size
        freq = args2.freq
        # Data is overwritten here!!
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args2.batch_size  # bsz for train and valid
        freq = args2.freq

    if args2.data == 'air':
        assert args2.task_name == 'imputation'
        drop_last = False
        # # PM25_Dataset
        # data_set = Data(root_path=args2.root_path, mode=flag, eval_length=args2.seq_len, target_dim=36, validindex=0)
        data_set = Dataset_Custom(
            root_path=args2.root_path,
            data_path=args2.data_path,
            flag=flag,
            size=[args2.seq_len, args2.label_len, args2.pred_len],
            features=args2.features,
            target=args2.target,
            timeenc=timeenc,
            freq=freq,
            AirDataset=True
        )
    elif args2.data == 'physio':
        assert args2.task_name == 'imputation'
        drop_last = False
        # real_Physio_Dataset
        # data_set = Data(mode=flag, seed=1, nfold=4)
        data_set = Dataset_Custom(
            root_path=args2.root_path,
            data_path=args2.data_path,
            flag=flag,
            size=[args2.seq_len, args2.label_len, args2.pred_len],
            features=args2.features,
            target=args2.target,
            timeenc=timeenc,
            freq=freq,
            Physio=True
        )
    elif args2.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args2.root_path,
            data_path=args2.data_path,
            flag=flag,
            size=[args2.seq_len, args2.label_len, args2.pred_len],
            features=args2.features,
            target=args2.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args2.seasonal_patterns
        )

    else:
        data_set = Data(
            root_path=args2.root_path,
            data_path=args2.data_path,
            flag=flag,
            size=[args2.seq_len, args2.label_len, args2.pred_len],
            features=args2.features,
            target=args2.target,
            timeenc=timeenc,
            freq=freq,
            train_ratio=args2.train_ratio,
            train_part_first=args2.train_part_first
        )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args2.num_workers,
        drop_last=drop_last)

    return data_set, data_loader
