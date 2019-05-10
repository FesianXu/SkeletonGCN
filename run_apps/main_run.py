# !/usr/bin/env python
# -*- coding:utf-8 -*-

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import time
import os
from collections import OrderedDict

import yaml 

from utils import parser
from functools import wraps
from utils import bars

def funcTitle(title, isTimeit=False):
    def inner_wrapper(func):
        @wraps(func)
        def print_title(*args, **kwargs):
            print(' ======================== SPLIT =========================')
            print('Loading the {}'.format(title))
            begin = time.clock()
            func(*args, **kwargs)
            if isTimeit:
                end = time.clock()
                print('Loaded the {} in {:0>2f} seconds'.format(title, end-begin))
            else:
                print('Loaded the {}'.format(title))
            print(' ======================== SPLIT =========================')
        return print_title
    return inner_wrapper

def import_class(name):
    '''
    dynamic import the packages
    '''
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class Processor(object):
    '''
    Include the training and evaluation model, dynamic loading the model
    '''
    def __init__(self, arg):
        assert arg.phase in ('train', 'test')

        self.arg = arg
        self._save_arg()
        self._load_data()
        self._load_model()
        self._load_optimizer()
        
    
    @funcTitle('Data Module', isTimeit=True)
    def _load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_workers,
                drop_last=True
            )
        
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.batch_size,
            shuffle=False,
            num_workers=self.arg.num_workers,
            drop_last=True
        )
    
    @funcTitle('Model Module', isTimeit=True)
    def _load_model(self):
        # only single GPU available
        output_device = self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args, 
                           batch_size=self.arg.batch_size,
                           device=self.arg.device).cuda(output_device)
        self.loss_fn = nn.CrossEntropyLoss().cuda(output_device)

        if self.arg.weights:
            self._load_weight()
        

    @funcTitle('Weights Module', isTimeit=True)
    def _load_weight(self):
        print('Loading weights from: {}'.format(self.arg.weights))
        checkpoint = torch.load(self.arg.weights)
        weights = checkpoint['weights']
        train_acc = checkpoint['train_acc']
        train_loss = checkpoint['train_loss']

        print(' => the train acc of the loaded weights : {}'.format(train_acc))
        print(' => the train loss of the loaded weights : {}'.format(train_loss))

        weights = OrderedDict([
            (u.split('module.')[-1], v.cuda(self.output_device)) for u, v in weights.items()
        ])

        try:
            self.model.load_state_dict(weights)
        except:
            state = self.model.state_dict()
            diff = list(
                set(state.keys()).difference(set(weights.keys()))
                )
            print('Cannot find these weights:')
            for d in diff:
                print('  ==> '+d)
            state.update(weights) # clean the unnecesary weights' keys
            self.model.load_state_dict(state)

    
    @funcTitle('Optimizer Module', isTimeit=True)
    def _load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    @funcTitle('Arg Saving Module', isTimeit=True)
    def _save_arg(self):
        '''
        save the arg in the work dir
        '''
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def _adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def _print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self._print_log("Local current time :  " + localtime)

    def _print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            if not os.path.exists(self.arg.work_dir):
                os.makedirs(self.arg.work_dir)
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def _record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def _split_time(self):
        split_time = time.time() - self.cur_time
        self._record_time()
        return split_time

    
    def start(self):
        if self.arg.phase == 'train':
            for each in (vars(self.arg)):
                self._print_log(' --> {}: {}'.format(each, vars(self.arg)[str(each)]))
            self._print_log(' =================> Here we go <===============')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)
                eval_model = ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)

                self._train(epoch, save_model=save_model)

                if eval_model:
                    with torch.no_grad():
                        self._eval(epoch, loader_name=['test'])
                else:
                    pass

        elif self.arg.phase == 'test':
            if self.arg.weights is None and not self.arg.force_run:
                raise ValueError('Please appoint --weights.')
            print('Model:   {}.'.format(self.arg.model))
            print('Weights: {}.'.format(self.arg.weights))
            with torch.no_grad():
                self._eval(epoch=0, loader_name=['test'], islog=False)
            print('Done.\n')

    @funcTitle('Save Model Module', isTimeit=True)
    def _save_model(self, model, model_path, **kwargs):
        state_dict = model.state_dict()
        weights = OrderedDict([
            (k.split('module.')[-1], v.cpu()) for k,v in state_dict.items()
        ])
        save_dict = {
            'weights': weights,
            'train_acc': kwargs['train_acc'],
            'train_loss': kwargs['train_loss']
        }
        torch.save(save_dict, model_path)
        
    
    def _train(self, epoch, save_model=False):
        self.model.train()
        self._print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        lr = self._adjust_learning_rate(epoch)
        loss_value = []
        acc_value = []
        acc_list = []

        tmp_loss_value = []
        tmp_acc_value = []

        self._record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        for batch_idx, (features, labels) in enumerate(loader):
            features = features.float().cuda(self.arg.device)
            labels = labels.cuda(self.arg.device)
            timer['dataloader'] += self._split_time()

            logits = self.model(features)
            lossv = self.loss_fn(logits, labels)

            pred = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
            result = (pred == labels).data.cpu().numpy()
            
            self.optimizer.zero_grad()
            lossv.backward()
            self.optimizer.step()
            loss_value.append(lossv.data.item())
            tmp_loss_value.append(lossv.data.item())

            acc_value.append(np.sum(result))
            tmp_acc_value.append(np.sum(result))
            timer['model'] += self._split_time()
            

            # statistics
            if batch_idx % self.arg.log_interval == 0 and batch_idx != 0:
                self._print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}, acc:{:.3f}'.format(
                        batch_idx, len(loader), sum(tmp_loss_value)/self.arg.log_interval, lr, \
                        sum(tmp_acc_value)/self.arg.log_interval/self.arg.batch_size))
                acc_list.append(sum(tmp_acc_value)/self.arg.log_interval/self.arg.batch_size )
                tmp_acc_value = []
                tmp_loss_value = []
            timer['statistics'] += self._split_time()
            
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self._print_log(
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self._print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))

        # save the model weights
        if save_model:
            model_save_path = '{}/epoch{}_model.pt'.format(self.arg.work_dir, epoch+1)
            self._save_model(model=self.model,
                             model_path=model_save_path,
                             train_loss=np.mean(loss_value),
                             train_acc=None if len(acc_list) is 0 else sum(acc_list)/len(acc_list))


    def _eval(self, epoch, loader_name=['test'], islog=True):
        self.model.train()
        if islog:
            self._print_log('Eval epoch: {}'.format(epoch + 1))

        for ln in loader_name:
            loss_value = []
            score_frag = []
            labels_list = []
            for batch_idx, (features, labels) in enumerate(self.data_loader[ln]):
                bars.print_toolbar(batch_idx * 1.0 / len(self.data_loader[ln]),
                '({:>5}/{:<5})'.format(batch_idx + 1, len(self.data_loader[ln])))
                
                features = features.float().cuda(self.arg.device)
                labels = labels.cuda(self.arg.device)
                
                logits = self.model(features)
                lossv = self.loss_fn(logits, labels)

                pred = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
                result = (pred == labels).data.cpu().numpy()

                score_frag.append(logits.data.cpu().numpy())
                labels_list.append(labels.data.cpu().numpy())
                loss_value.append(lossv.data.item())
            bars.end_toolbar()

            score = np.concatenate(score_frag)
            labels_np = np.concatenate(labels_list)
            predict = np.argmax(score, axis=-1)
            acc = np.sum(np.equal(labels_np, predict).astype(np.int32))/predict.shape[0]
            if islog:
                self._print_log('\tMean {} loss of {} batches: {}.'.format(
                    ln, len(self.data_loader[ln]), np.mean(loss_value)))
                self._print_log('The accuracy over {} samples is {}'.format(predict.shape[0], acc))
            else:
                print('\tMean {} loss of {} batches: {}.'.format(
                    ln, len(self.data_loader[ln]), np.mean(loss_value)))
                print('The accuracy over {} samples is {}'.format(predict.shape[0], acc))
                


if __name__ == '__main__':
    parserfn = parser.get_parser()
    
    # load arg form config file
    print('Begin here')
    p = parserfn.parse_args()

    if p.config is not None:
        # parse the yaml
        with open(p.config, 'r') as f:
            default_args = yaml.load(f)
        key = vars(p).keys()

        for k in default_args.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        
        parserfn.set_defaults(**default_args)
    # load the default parameters from yaml files

    arg = parserfn.parse_args()
    processor = Processor(arg)
    processor.start()