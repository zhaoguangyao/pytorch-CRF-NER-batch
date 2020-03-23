# -*- coding: utf-8 -*-
import os
import sys
import time
import torch
import subprocess
import numpy as np

import torch.optim as optim
from driver.Loader import create_batch_iter, pair_data_variable


def train(model, train_data, dev_data, test_data, vocab_srcs, vocab_tgts, config):
    # optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if config.learning_algorithm == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.weight_decay)
    elif config.learning_algorithm == 'adam':
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise RuntimeError('Invalid optimizer method: ' + config.learning_algorithm)

    # train
    global_step = 0
    best_acc = 0
    print('\nstart training...')
    for iter in range(config.epochs):
        iter_start_time = time.time()
        print('Iteration: ' + str(iter))

        batch_num = int(np.ceil(len(train_data) / float(config.batch_size)))
        batch_iter = 0
        for batch in create_batch_iter(train_data, config.batch_size, shuffle=True):
            start_time = time.time()
            batch, feature, target, lengths, mask = pair_data_variable(batch, vocab_srcs, vocab_tgts, config.use_cuda)
            model.train()
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(feature, target, lengths, mask)
            loss_value = loss.item()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimizer.step()

            accuracy = evaluate_batch(model, batch, feature, lengths, mask, vocab_tgts, config)

            during_time = float(time.time() - start_time)
            print("Step:{}, Iter:{}, batch:{}, accuracy:{:.4f}, time:{:.2f}, loss:{:.6f}"
                  .format(global_step, iter, batch_iter, accuracy, during_time, loss_value))

            batch_iter += 1
            global_step += 1

            if batch_iter % config.test_interval == 0 or batch_iter == batch_num:
                if dev_data is not None:
                    dev_acc = evaluate(model, dev_data, global_step, vocab_srcs, vocab_tgts, "dev", config)
                test_acc = evaluate(model, test_data, global_step, vocab_srcs, vocab_tgts, "test", config)
                if dev_data is not None:
                    if dev_acc > best_acc:
                        print("Exceed best acc: history = %.2f, current = %.2f" % (best_acc, dev_acc))
                        best_acc = dev_acc
                    if -1 < config.save_after <= iter:
                        torch.save(model.state_dict(), os.path.join(config.model_path, 'model.' + str(global_step)))
                else:
                    if test_acc > best_acc:
                        print("Exceed best acc: history = %.2f, current = %.2f" % (best_acc, test_acc))
                        best_acc = test_acc
                        if -1 < config.save_after <= iter:
                            torch.save(model.state_dict(), os.path.join(config.model_path, 'model.' + str(global_step)))
        during_time = float(time.time() - iter_start_time)
        print('one iter using time: time:{:.2f}'.format(during_time))


def evaluate_batch(model, batch, feature, lengths, mask, vocab_tgts, config):

    _, tags = model(feature, lengths, mask)

    # 输出到文件
    ori_path = os.path.join(config.model_path, "batch.txt")
    with open(ori_path, 'w', encoding='utf-8') as output_file:
        for idx, seq in enumerate(batch):
            for idj in range(len(batch[idx][0])):
                output_file.write(batch[idx][0][idj] + " ")
                output_file.write(batch[idx][1][idj] + " ")
                output_file.write(vocab_tgts.id2word(tags[idx][idj].item()) + "\n")
            output_file.write('\n')
    m_system = sys.platform
    unix_path = os.path.join(config.model_path, 'unix_batch.txt')
    if m_system == 'win32':
        sp = subprocess.check_call("perl -p -e 's/\\r$//' < " + ori_path + " > " + unix_path, shell=True)
        assert sp == 0
    elif m_system == 'linux':
        sp = subprocess.check_call('cp ' + ori_path + ' ' + unix_path, shell=True)
        assert sp == 0
    else:
        print('没有使用过这个系统')
    perl_res = subprocess.check_output("perl ./driver/conlleval.pl < " + unix_path, shell=True)
    output = perl_res.decode("utf-8")
    line2 = output.split('\n')[1]
    fours = line2.split(';')
    accuracy = float(fours[0][-7:-1])
    return accuracy


def evaluate(model, data, step, vocab_srcs, vocab_tgts, dev_test, config):
    model.eval()

    ori_path = os.path.join(config.model_path, dev_test + "_ori_" + str(step) + ".txt")
    with open(ori_path, 'w', encoding='utf-8') as output_file:
        for batch in create_batch_iter(data, config.batch_size):
            batch, feature, target, lengths, mask = pair_data_variable(batch, vocab_srcs, vocab_tgts, config)
            _, tags = model(feature, lengths, mask)

            # 输出到文件
            for idx, seq in enumerate(batch):
                for idj in range(len(batch[idx][0])):
                    output_file.write(batch[idx][0][idj] + " ")
                    output_file.write(batch[idx][1][idj] + " ")
                    output_file.write(vocab_tgts.id2word(tags[idx][idj].item()) + "\n")
                output_file.write("\n")

    m_system = sys.platform
    unix_path = os.path.join(config.model_path, dev_test + "_unix_" + str(step) + ".txt")
    if m_system == 'win32':
        sp = subprocess.check_call("perl -p -e 's/\\r$//' < " + ori_path + " > " + unix_path, shell=True)
        assert sp == 0
        os.remove(ori_path)
    elif m_system == 'linux':
        sp = subprocess.check_call('mv ' + ori_path + ' ' + unix_path, shell=True)
        assert sp == 0
    else:
        print('没有使用过这个系统')

    p = subprocess.check_output("perl ./driver/conlleval.pl < " + unix_path, shell=True)
    output = p.decode("utf-8")
    line2 = output.split('\n')[1]
    fours = line2.split(';')
    accuracy = float(fours[0][-7:-1])
    precision = float(fours[1][-7:-1])
    recall = float(fours[2][-7:-1])
    f1 = float(fours[3][-7:-1])

    print('accuracy: {:.4f} precision: {:.4f}% recall: {:.4f}% f1: {:.4f}% \n'.format(accuracy, precision,
                                                                                      recall, f1))

    model.train()
    return accuracy
