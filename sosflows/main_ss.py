import argparse
from utils import *
# edited by Jingjing
from datasets import *

if __name__ == '__main__':

    if sys.version_info < (3, 6):
        print('Sorry, this code might need Python 3.6 or higher')

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Flows')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='input batch size fcd or training (default: 100)')
    parser.add_argument(
        '--test-batch-size',
        type=int,
        default=1000,
        help='input batch size for testing (default: 1000)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='number of epochs to train (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
    parser.add_argument(
        '--dataset',
        default='POWER',
        help='POWER | GAS | HEPMASS | MINIBONE | BSDS300 | MOONS')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--num-blocks',
        type=int,
        default=5,
        help='number of invertible blocks (default: 5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1000,
        help='how many batches to wait before logging training status')
    parser.add_argument(
        '--K',
        type=int,
        default=5,
        help='number of polynomials to use')
    parser.add_argument(
        '--M',
        type=int,
        default=3,
        help='degree of polynomials to use')
    parser.add_argument(
        '--name',
        type=str,
        default='',
        help='run name')
    parser.add_argument(
        '--mode',
        type=str,
        default='direct',
        help='mode')

    args = parser.parse_args()

    # CUDA
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # manual seed
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    assert args.dataset in ['POWER', 'GAS', 'HEPMASS', 'MINIBONE', 'BSDS300', 'MOONS']
    
    dataset = getattr(datasets, args.dataset)()

    
    #
    train_dataset, valid_dataset, test_dataset = make_datasets(dataset.trn.x, dataset.val.x, dataset.tst.x)
    #

    train_loader, valid_loader, test_loader = make_loaders(train_dataset, valid_dataset, test_dataset,
                                                          args.batch_size, args.test_batch_size, **kwargs)
    print('Train_Batch_Num:'.format(len(train_loader)))
    print('Valid_Batch_Num:'.format(len(valid_loader)))
    print('Test_Batch_Num:'.format(len(test_loader)))
    
    print("Prepare Data from "+args.dataset)
   
    num_inputs = dataset.n_dims       
    num_hidden = {
        'POWER': 100,
        'GAS': 100,
        'HEPMASS': 512,
        'MINIBOONE': 512,
        'BSDS300': 512,
        'MOONS': 64
    }[args.dataset]

    model, optimizer = build_model(args.num_blocks, num_inputs, num_hidden, args.K, args.M, args.lr, device=device)
    
    best_model_forward, test_loss_forward = train(model, optimizer, train_loader, valid_loader, test_loader,
                                                  args.epochs, device, args.log_interval, True, args.mode)

    '''
    train_dataset_inv, valid_dataset_inv, test_dataset_inv = make_inverse_datasets(best_model_forward, train_dataset,

    train_loader_inv, valid_loader_inv, test_loader_inv = make_loaders(train_dataset_inv, valid_dataset_inv,test_dataset_inv,args)

    best_model, test_loss_inv = train(best_model_forward, optimizer, train_loader_inv, valid_loader_inv, args.epochs, device, 'inverse')
    '''

    args.name = args.dataset
    name = args.name if len(args.name) > 0 else default_name()
    path = MODEL_DIR + name

    save_dict = {
        'model': best_model_forward,
        'optim': optimizer,
        'args': args,
        'test_loss': test_loss_forward
    }
    torch.save(save_dict, path)

    '''
    if args.dataset == 'MOONS':
        # generate some examples
        best_model.eval()
        u = np.random.randn(500, 2).astype(np.float32)
        u_tens = torch.from_numpy(u).to(device)
        x_synth = best_model.forward(u_tens, mode='inverse')[0].detach().cpu().numpy()
    
        import matplotlib.pyplot as plt
    
        fig = plt.figure()
    
        ax = fig.add_subplot(121)
        ax.plot(dataset.val.x[:,0], dataset.val.x[:,1], '.')
        ax.set_title('Real data')
    
        ax = fig.add_subplot(122)
        ax.plot(x_synth[:,0], x_synth[:,1], '.')
        ax.set_title('Synth data')
    
        plt.show()
    '''