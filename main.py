import argparse
import numpy as np
import math
import timeit
import torch
from torchvision.transforms import ToTensor
from torch.autograd import Variable

from skvideo.io import VideoCapture
import matplotlib.pyplot as plt

from sensory.visual import VisualNetwork
from temporary.con import Recog


def np_to_tensor(nparray):
    x = image.astype('float')
    x.transpose((2, 0, 1))
    x = ToTensor()(x)

    x -= 0.5
    return x.cuda()


def visualize_feats(feats, n=256):
    s1, s2 = feats.shape[-2:]  # feature map size
    nsq = math.floor(math.sqrt(n))

    composed = np.zeros((nsq * s1, nsq * s2))
    for iy in range(nsq):
        for ix in range(nsq):
            composed[iy*s1:(iy+1)*s1, ix*s2:(ix+1)*s2] = feats[ix+iy*nsq, ...]

    att = np.squeeze(np.sum(feats ** 2, axis=0) / feats.shape[0])

    return composed, att

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments to wake Tylor.')
    parser.add_argument(
        '-f',
        '--file',
        type=str,
        default=None,
        required=True,
        help='The input file. A local path or an youtube url')
    parser.add_argument(
        '--skip',
        type=int,
        default=5,
        help='args.skip every N frames'
    )
    parser.add_argument(
        '--buffsz',
        type=int,
        default=4,
        help='frame buffer size'
    )
    parser.add_argument(
        '--nfeat2show',
        type=int,
        default=64
    )
    parser.add_argument('-v', '--view', dest='view', action='store_true')
    parser.set_defaults(view=False)
    args = parser.parse_args()

    # Define Visual Network
    visual_sensor = VisualNetwork()
    visual_sensor = visual_sensor.cuda()

    # Define Temporal Network
    lstm = Recog(2048, 512, 128)
    lstm = lstm.cuda()

    # Open Video
    cap = VideoCapture(args.file)
    cap.open()

    f1, ax1 = plt.subplots(1, 3)
    f2, ax2 = plt.subplots(2, 1)

    plt.ion()

    image_buffer = [0]*args.buffsz
    buffer = None

    FPS = 0
    t_buffer_start = timeit.default_timer()

    bufferidx = 0
    frameidx = 0
    while True:
        print('frame:', frameidx, 'FPS:', FPS)
        retval, image = cap.read()

        if not retval:
            break

        if buffer is None:
            buffer = torch.FloatTensor(args.buffsz, 3,
                                       image.shape[0], image.shape[1]).cuda()

        if frameidx % args.skip is not 0:
            frameidx += 1
            continue
        else:
            frameidx += 1
            # Fill the buffer
            x = np_to_tensor(image)
            image_buffer[bufferidx] = image
            buffer[bufferidx, :, :, :] = x
            bufferidx += 1

        if bufferidx < args.buffsz:
            continue

        fpool, f4 = visual_sensor(Variable(buffer, volatile=True))

        if args.view:
            vcode = fpool.data[:, :args.nfeat2show, :, :].cpu()  # For inspection
            f4 = f4.data[:, :args.nfeat2show, :, :].cpu().numpy()  # For inspection
            
            for b in range(args.buffsz):
                plt.pause(0.0001) 
                lstm_out = lstm(fpool[b, ...].view(1, 1, -1))
                lstm_out = lstm_out.data[0, :args.nfeat2show].cpu().numpy()  # For inspection
                
                ax1[0].cla()
                ax1[1].cla()
                ax1[2].cla()
                ax2[0].cla()
                ax2[1].cla()
                frame = image_buffer[b]
                code = vcode[b, :, :, :].squeeze().cpu().numpy()

                try:
                    ax1[0].imshow(frame)
                except TypeError:
                    print(frame)

                ax1[1].plot(code)

                FPS = 1 / ((timeit.default_timer() - t_buffer_start) / frameidx)
                ax1[0].text(40, 40, 'FPS:%.3f' % FPS, color='b')
                
                ax1[2].plot(lstm_out)            
                feats2show, att = visualize_feats(f4[b, ...], args.nfeat2show)
                ax2[0].imshow(feats2show, cmap='hot')
                ax2[1].imshow(att, cmap='hot')
                
                f1.show()
                f2.show()
                f1.canvas.draw()
                f2.canvas.draw() 
        else:
            FPS = 1 / ((timeit.default_timer() - t_buffer_start) / frameidx)
            print('FPS:', FPS)
                
        # Clear the buffer
        bufferidx = 0
        image_buffer = [0] * args.buffsz
        buffer = torch.FloatTensor(args.buffsz, 3, image.shape[0], image.shape[1]).cuda()

    print('Done')