import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.noisedata import NoiseData
from utils.transform import Normalizer
from model.nonlinear import NonLinear, NonLinearType, NonLinearTypeModel, NonLinearBowlMode
import torch
from torch.autograd import Variable
from torch import nn

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Noise estimation')
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=4, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='../data', type=str)
    parser.add_argument('--filename', dest='filename', help='data filename.',
          default='data_final_test.xlsx', type=str)
    parser.add_argument('--dataset', dest='dataset', help='Dataset type.', default='NoiseData', type=str)
    parser.add_argument('--snapshot', dest='snapshot', help='Name of model snapshot.',
          default='', type=str)
    parser.add_argument('--nc', dest='nc', type = int, default = 3200)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    batch_size = args.batch_size
    snapshot_path = args.snapshot
    transformations = Normalizer(mean=[362.69, 60.67, 2372.96, 149.45, 67.89, 7.65], std=[130.04, 209.28, 930.67, 5.79, 6.88, 0.10])

    if args.dataset == 'NoiseData':
        dataset = NoiseData(dir=args.data_dir, filename='data_final_test_0318.xlsx', transform=transformations, use_type=True)

    print ('Loading snapshot.')
    # Load snapshot
    model = NonLinearBowlMode(nc=args.nc).to(device)
    saved_state_dict = torch.load(snapshot_path, map_location=device)
    model.load_state_dict(saved_state_dict)
    
    test_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2)
    
    criterion = nn.MSELoss().to(device)
    test_error = .0
    total = 0

    for i, (inputs, outputs, types, bowl, sheet_idx) in tqdm(enumerate(test_loader)):
        total += outputs.size(0)
        inputs = inputs.to(device)
        labels = outputs.to(device)
        bowl = bowl.long().to(device)
        sheet_idx = sheet_idx.to(device)
        
        preds = model(inputs)
        
        batch_indices = torch.arange(preds.size(0), device=device)
        preds = preds[batch_indices, sheet_idx.squeeze(), :]
        bowl = bowl.view(-1, 1)
        preds = preds.gather(1, bowl)
       
        test_loss = criterion(preds, labels)
        test_error += torch.sum(test_loss)
        # print(preds, labels, test_loss, torch.sum(test_loss))
    
    print('Test error on the ' + str(total) +' test samples. MSE: %.4f' % (test_error * batch_size/ total))