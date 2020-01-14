#
#  dataset.py
#  Hopkins
#
#  Created by Rolan Akhmedov on 04.07.18.
#  Copyright © 2018 Rolan Akhmedov. All rights reserved.
#

import json
import numpy as np
import matplotlib.pyplot as plot


# def color(char_id):
# 	return {
# 		0: 'white',
# 		1: 'blue',
# 		2: 'green',
# 		3: 'black',
# 		4: 'yellow'
# 	}[char_id]


class Dataset:

    @staticmethod
    def check_balance(filename):
        file = open(filename)
        dataset = json.load(file)
        statistic = dataset['radix'] * [0.0]
        for item in dataset['dataset']:
            statistic[item['annotation']['signal_id']] += 1.0
        return 100 * np.array(statistic) / dataset['items_in_dataset']

    @staticmethod
    def check_discret_rate(filename):
        file = open(filename)
        dataset = json.load(file)['dataset']
        rate = [(item['annotation']['time_to'] - item['annotation']['time_from'])
                for item in dataset if item['annotation']['signal_id']]
        plot.hist(rate, bins=30)
        plot.ylabel('Discretization rate')
        plot.show()


    @staticmethod
    def snr_statistic(filename):
        file = open(filename)
        dataset = json.load(file)

        snr = [item['annotation']['SNR'] for item in dataset['dataset']
               if item['annotation']['SNR'] and item['annotation']['signal_id'] != 2]

        plot.hist(snr, bins=30)
        plot.ylabel('Probability')
        plot.show()

        return min(snr), sum(snr)/len(snr), max(snr)

    @staticmethod
    def sequence_to_label(filename):
        file = open(filename)
        dataset = json.load(file)

        label = np.zeros((dataset['items_in_dataset'], dataset['radix']), dtype=float)
        for ds_item, lb_item in zip(dataset['dataset'], label):
            lb_item[ds_item['annotation']['signal_id']] = 1

        field = [item['field'] for item in dataset['dataset']]
        return np.array(field, dtype=float), label

    @staticmethod
    def sequence_to_sequence(filename):
        file = open(filename)
        data = json.load(file)

        ground_truth = np.zeros((data['items_in_dataset'], data['radix'], data['item_length']), dtype=float)
        for ds_item, gt_item in zip(data['dataset'], ground_truth):
            label = ds_item['annotation']['signal_id']
            time_from = ds_item['annotation']['time_from']
            time_to = ds_item['annotation']['time_to']
            gt_item[label][time_from:time_to] = np.full((time_to-time_from), 1, dtype=float)

        field = [item['field'] for item in data['dataset']]
        return np.array(field, dtype=float), ground_truth


def main():
    X, Y = Dataset.sequence_to_sequence('lira.json')
    print('Class balance: ', Dataset.check_balance('lira.json'))
    Dataset.snr_statistic('lira.json')
    # Dataset.check_discret_rate('lira.json')

    for data, output in zip(X[:20], Y[:20]):
        plot.plot(data, label='Ex')
        plot.plot([50 * f for f in output[1]], label='S1')
        plot.plot([50 * f for f in output[2]], label='S2')
        plot.plot([50 * f for f in output[3]], label='S3')
        plot.legend()
        plot.show()


if __name__ == '__main__':
    main()
