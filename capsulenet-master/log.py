import logging
import visdom
import torch
from PIL import Image
from os import path


class Logger(object):
    def __init__(self, batch_size, use_visdom, img_dir='./generated'):
        self.use_visdom = use_visdom
        self.viz = visdom.Visdom()
        self.batch_size = batch_size
        self.img_dir = img_dir

        log_handle = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        log_handle.setFormatter(formatter)

        log = logging.getLogger()
        log.addHandler(log_handle)
        log.setLevel(logging.INFO)
        self.__logger = log

        if use_visdom:
            # Track the visdom panes
            self.panes = {}
        self.offsets = {'imgs': 0}

    def plain(self, msg):
        self.__logger.info(msg)

    def log(self, epoch, batch_index, data_len, ctx, **kwargs):
        '''
        Logs metrics to stdout and visdom
        params:
        - epoch: training iteration
        - batch_index: nth item in the batch
        - data_len: length of the dataset
        - ctx: what is being logged, typically train, test, or validate
        - kwargs: the scalar values to log
        '''

        msg = '{:<6} [Epoch {: <2} {:2.0f}%] '
        keys = []
        for key in kwargs:
            msg += key + ': {:.3f} '
            keys.append(key)  # want to ensure the order of items in the list dict iteration can differ

        n_batches = int(data_len / self.batch_size)
        n_batches = 1 if n_batches == 0 else n_batches  # dont divide by 0

        self.__logger.info(msg.format(
            ctx,
            epoch,
            int((batch_index / n_batches)*100),
            *[kwargs[key] for key in keys]
        ))

        if self.use_visdom:
            for key, value in kwargs.items():
                pane = ctx + '_' + key
                if pane in self.panes:
                    X = torch.Tensor([self.offsets[pane]])
                    Y = torch.Tensor([value])
                    self.panes[pane] = self.viz.line(X=X, Y=Y,
                                                     win=self.panes[pane],
                                                     update='append',
                                                     opts={'title': pane})
                    self.offsets[pane] += 1

                else:
                    X, Y = torch.Tensor([0]), torch.Tensor([value])
                    self.panes[pane] = self.viz.line(X=X, Y=Y,
                                                     opts={'title': pane})
                    self.offsets[pane] = 1

    def images(self, ctx, X):
        '''Shows a batch of images in visdom'''
        # self.viz.images(X)
        img = self.gallery(X, 8)
        img = img.view(*img.shape[1:]).numpy() * 255

        im = Image.fromarray(img).convert('RGB')
        im.save(path.join(self.img_dir,
                          '{}_{}.png'.format(ctx, self.offsets['imgs'])))
        self.offsets['imgs'] += 1

    def gallery(self, array, ncols=5):
        nindex = array.shape[0]
        nrows = nindex//ncols
        assert nindex == nrows*ncols
        # want result.shape = (height*nrows, width*ncols, intensity)
        rows = []
        for i in range(nrows):
            start = i * ncols
            end = (i+1) * ncols
            rows.append(torch.cat(array[start:end], dim=1))
        return torch.cat(rows, dim=2)
