from argparse import ArgumentParser
import os
import torch
import punet
import parc

# print information
print('\n')
print('ParcNet cortical parcellation')
print('\n')

# parse arguments
parser = ArgumentParser()
parser.add_argument("path_images", type=str, help="images to super-resolve / synthesize. Can be the path to a single image or to a folder")
parser.add_argument("path_predictions", type=str,
                    help="path where to save the synthetic 1mm MP-RAGEs. Must be the same type "
                         "as path_images (path to a single image or to a folder)")
parser.add_argument("--model", type=str, help="path to saved weightts")
parser.add_argument("--cpu", action="store_true", help="enforce running with CPU rather than GPU.")
args = vars(parser.parse_args())

# enforce CPU processing if necessary
if args['cpu']:
    print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Prepare list of images to process
path_images = os.path.abspath(args['path_images'])
basename = os.path.basename(path_images)
path_predictions = os.path.abspath(args['path_predictions'])

dataset = parc.PARC(root=path_images, subset='.', split=None, mode='image', labels='', in_channels=3, num_classes=32, labeled=False)
model = punet.unet2d_320_dktatlas_positional_20_1_0_0(loadpath=args["model"]).to(device)
percentile = 0.02

print('Found %d subjects' % len(dataset))
for idx in range(len(dataset)):
    print('  Working on subject %d ' % (idx+1))
    img = dataset.__getitem__(idx)[0].to(device)
    minvals = torch.kthvalue(img.flatten(1),round(img.flatten(1).shape[1]*(percentile-0)),dim=1)[0][:,None,None]
    maxvals = torch.kthvalue(img.flatten(1),round(img.flatten(1).shape[1]*(1-percentile)),dim=1)[0][:,None,None]
    img = torch.min(torch.max(img,minvals),maxvals)
    img = (img - img.flatten(1).mean(1).reshape(3,1,1)) * (1 / img.flatten(1).std(1).reshape(3,1,1))

    dataset.save_output(path_predictions, [model(img[None]).detach().cpu()[0].argmax(0, keepdims=True)], [idx])

print(' ')
print('All done!')
print(' ')

