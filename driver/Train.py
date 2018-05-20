# -*- coding: utf-8 -*-
import os
import time
import torch
import numpy
import torch.optim as optim
from driver.DataLoader import create_batch_iter, pair_data_variable


def train(model, train_data, dev_data, test_data, vocab_srcs, vocab_tgts, config):
    model.train()
    # optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    if config.learning_algorithm == 'sgd':
        optimizer = optim.SGD(parameters, lr=config.lr, weight_decay=config.weight_decay)
    elif config.learning_algorithm == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=config.lr, weight_decay=config.weight_decay)
    elif config.learning_algorithm == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=config.lr, weight_decay=config.weight_decay)
    elif config.learning_algorithm == 'adam':
        optimizer = optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise RuntimeError("Invalid optim method: " + config.learning_algorithm)

    # train
    global_step = 0
    best_f1 = 0
    print('\nstart training...')
    for iter in range(config.epochs):
        iter_start_time = time.time()
        print('Iteration: ' + str(iter))

        batch_num = int(numpy.ceil(len(train_data) / float(config.batch_size)))
        batch_iter = 0
        for batch in create_batch_iter(train_data, config.batch_size, shuffle=True):
            start_time = time.time()
            feature, target, feature_lengths = pair_data_variable(batch, vocab_srcs, vocab_tgts, config)

            optimizer.zero_grad()
            h_output = model(feature, feature_lengths)
            loss = model.get_loss(h_output, feature_lengths, target)
            loss_value = loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()

            during_time = float(time.time() - start_time)
            print("Step:{}, Iter:{}, batch:{}, time:{:.2f}, loss:{:.6f}"
                  .format(global_step, iter, batch_iter, during_time, loss_value[0]))

            batch_iter += 1
            global_step += 1

            if batch_iter % config.test_interval == 0 or batch_iter == batch_num:
                if config.dev_file:
                    dev_f1 = evaluate(model, dev_data, global_step, vocab_srcs, vocab_tgts, config)
                if config.test_file:
                    test_f1 = evaluate(model, test_data, global_step, vocab_srcs, vocab_tgts, config)
                if config.dev_file:
                    if dev_f1 > best_f1:
                        print("Exceed best acc: history = %.2f, current = %.2f" % (best_f1, dev_f1))
                        best_f1 = dev_f1
                        if os.path.exists(config.save_model_path):
                            pass
                        else:
                            os.makedirs(config.save_model_path)
                        if -1 < config.save_after <= iter:
                            torch.save(model.state_dict(), os.path.join(config.save_model_path,
                                                                        'model.' + str(global_step)))
                else:
                    if test_f1 > best_f1:
                        print("Exceed best acc: history = %.2f, current = %.2f" % (best_f1, test_f1))
                        best_f1 = test_f1
                        if os.path.exists(config.save_model_path):
                            pass
                        else:
                            os.makedirs(config.save_model_path)
                        if -1 < config.save_after <= iter:
                            torch.save(model.state_dict(), os.path.join(config.save_model_path,
                                                                        'model.' + str(global_step)))
        during_time = float(time.time() - iter_start_time)
        print('one iter using time: time:{:.2f}'.format(during_time))


def evaluate(model, data, step, vocab_srcs, vocab_tgts, config):
    model.eval()
    start_time = time.time()
    predict_number, gold_number, correct_number = 0, 0, 0

    for batch in create_batch_iter(data, config.batch_size):
        feature, label, feature_lengths = pair_data_variable(batch, vocab_srcs, vocab_tgts, config)

        h = model(feature, feature_lengths)
        predict = model._viterbi_decode(h, feature_lengths)
        label = label.view(len(predict))
        for idx, value in enumerate(predict):
            if label.data[idx] != vocab_tgts.word2id('O'):
                gold_number += 1
            if value == vocab_tgts.word2id('O'):
                continue
            elif value == label.data[idx]:
                predict_number += 1
                correct_number += 1
            else:
                predict_number += 1
    if predict_number == 0:
        p = 0
    else:
        p = correct_number / predict_number
    if gold_number == 0:
        r = 0
    else:
        r = correct_number / gold_number
    if (p + r) == 0:
        f_score = 0
    else:
        f_score = 2 * p * r / (p + r)
    during_time = float(time.time() - start_time)
    print("\nevaluate result: ")
    print("Step:{}, f1:{:.4f}, time:{:.2f}"
          .format(step, f_score, during_time))
    model.train()
    return f_score