import argparse
    
def load_config():

    # args
    parser = argparse.ArgumentParser(description="PialNN")
    
    # data
    parser.add_argument('--data_path', default="./data/train/", type=str, help="path of the dataset")
    parser.add_argument('--hemisphere', default="lh", type=str, help="left or right hemisphere (lh or rh)")
    # model file
    parser.add_argument('--model', help="path to best model")
    #model
    parser.add_argument('--nc', default=128, type=int, help="num of channels")
    parser.add_argument('--K', default=5, type=int, help="kernal size")
    parser.add_argument('--n_scale', default=3, type=int, help="num of scales for image pyramid")
    parser.add_argument('--n_smooth', default=1, type=int, help="num of Laplacian smoothing layers")
    parser.add_argument('--lambd', default=1.0, type=float, help="Laplacian smoothing weights")
    # training
    parser.add_argument('--train_data_ratio', default=0.8, type=float, help="percentage of training data")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--n_epoch', default=200, type=int, help="total training epochs")
    parser.add_argument('--ckpts_interval', default=10, type=int, help="save checkpoints after each n epoch")
    parser.add_argument('--report_training_loss', default=True, type=bool, help="if report training loss")
    parser.add_argument('--save_model', default=True, type=bool, help="if save training models")
    parser.add_argument('--save_mesh_train', default=False, type=bool, help="if save mesh during training")
    # evaluation
    parser.add_argument('--save_mesh_eval', default=False, type=bool, help="if save mesh during evaluation")
    parser.add_argument('--n_test_pts', default=150000, type=int, help="num of points sampled for evaluation")

    config = parser.parse_args()

    return config
