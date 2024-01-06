import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import torch

styles = ['.-r', '.--b', '.--g', '.--c', '.-m', '.-y', '.-k', '.-c', '.-c', '.-c', '.-c', '.-c', '.-c', '.-c', '.-c', '.-c', '.-c', '.-c', '.-c', '.-c']
#lines = ['-', '-.', '--', '--', '--', '--', '--', '--', '--', '--', ':', ':', ':', ':', ':', ':', ':']
lines = ['-', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':', ':']
points = ['*', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
colors = ['r', 'b', 'g', 'c', 'm', 'orange', 'k', 'navy', 'cyan', 'purple', 'orange', 'indianred', 'darkorange', 'darkgoldenrod', 'turquoise', 'orchid', 'aqua', 'coral', 'peru']


def main(cfg):
    method_names = cfg.methods.split('+')
    dataset_names = cfg.datasets.split('+')
    os.makedirs(cfg.out_dir, exist_ok=True)
    # plt.style.use('seaborn-white')

    # Plot PR Cureve
    for dataset in dataset_names:
        plt.figure()
        idx_style = 0
        for method in method_names:
            iRes = torch.load(
                os.path.join(cfg.res_dir, dataset + '_' + method + '.pth'))
            imax = np.argmax(iRes['Fm'])
            plt.plot(
                iRes['Prec'],
                iRes['Recall'],
                #  styles[idx_style],
                color=colors[idx_style],
                linestyle=lines[idx_style],
                marker=points[idx_style],
                markevery=[imax, imax],
                label=method)
            idx_style += 1

        plt.grid(True, zorder=-1)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1.02)
        plt.ylabel('Precision', fontsize=25)
        plt.xlabel('Recall', fontsize=25)

        plt.legend(
            ['Ours', 'R3Net', 'PoolNet', 'LVNet', 'EGNet', 'MINet', 'GateNet', 'U2Net', 'F3Net', 'DAFNet', 'SARNet',
                    'MJRBM', 'MSCNet', 'MCCNet', 'ERPNet', 'SH-Net', 'SeaNet', 'MEANet', 'BAFSNet'], loc='lower left', prop={'size': 10}, ncol=4)
        plt.savefig(os.path.join(cfg.out_dir, 'PR_' + dataset + '.png'),
                    dpi=600,
                    bbox_inches='tight')
        plt.close()

    # Plot Fm Cureve
    for dataset in dataset_names:
        plt.figure()
        idx_style = 0
        for method in method_names:
            iRes = torch.load(
                os.path.join(cfg.res_dir, dataset + '_' + method + '.pth'))
            imax = np.argmax(iRes['Fm'])
            plt.plot(
                np.arange(0, 255),
                iRes['Fm'],
                #  styles[idx_style],
                color=colors[idx_style],
                linestyle=lines[idx_style],
                marker=points[idx_style],
                label=method,
                markevery=[imax, imax])
            idx_style += 1
        plt.grid(True, zorder=-1)
        # plt.ylim(0, 1)
        plt.ylabel('F-measure', fontsize=25)
        plt.xlabel('Threshold', fontsize=25)

        plt.legend(
            ['Ours', 'R3Net', 'PoolNet', 'LVNet', 'EGNet', 'MINet', 'GateNet', 'U2Net', 'F3Net', 'DAFNet', 'SARNet',
                    'MJRBM', 'MSCNet', 'MCCNet', 'ERPNet', 'SH-Net', 'SeaNet', 'MEANet', 'BAFSNet'],loc='lower left', prop={'size': 10}, ncol=4)
        plt.savefig(os.path.join(cfg.out_dir, 'Fm_' + dataset + '.png'),
                    dpi=600,
                    bbox_inches='tight')
        plt.close()

    # Plot Em Cureve
    for dataset in dataset_names:
        plt.figure()
        idx_style = 0
        for method in method_names:
            iRes = torch.load(
                os.path.join(cfg.res_dir, dataset + '_' + method + '.pth'))
            imax = np.argmax(iRes['Em'])
            plt.plot(
                np.arange(0, 255),
                iRes['Em'],
                #  styles[idx_style],
                color=colors[idx_style],
                linestyle=lines[idx_style],
                marker=points[idx_style],
                label=method,
                markevery=[imax, imax])
            idx_style += 1
        plt.grid(True, zorder=-1)
        plt.ylim(0, 1)
        plt.ylabel('E-measure', fontsize=16)
        plt.xlabel('Threshold', fontsize=16)

        plt.legend(
            ['Ours', 'R3Net', 'PoolNet', 'LVNet', 'EGNet', 'MINet', 'GateNet', 'U2Net', 'F3Net', 'DAFNet', 'SARNet',
                    'MJRBM', 'MSCNet', 'MCCNet', 'ERPNet', 'SH-Net', 'SeaNet', 'MEANet', 'BAFSNet'],loc='best', prop={'size': 10}, ncol=4)
        plt.savefig(os.path.join(cfg.out_dir, 'Em_' + dataset + '.png'),
                    dpi=600,
                    bbox_inches='tight')
        plt.close()

    # Plot ROC Cureve
    for dataset in dataset_names:
        plt.figure()
        idx_style = 0
        for method in method_names:
            iRes = torch.load(
                os.path.join(cfg.res_dir, dataset + '_' + method + '.pth'))
            imax = np.argmax(iRes['Fm'])
            plt.plot(
                iRes['FPR'],
                iRes['TPR'],
                #  styles[idx_style][1:],
                color=colors[idx_style],
                linestyle=lines[idx_style],
                label=method)
            idx_style += 1

        plt.grid(True, zorder=-1)
        plt.xlim(0, 1)
        plt.ylim(0, 1.02)
        plt.ylabel('TPR', fontsize=16)
        plt.xlabel('FPR', fontsize=16)

        plt.legend(
            ['Ours', 'R3Net', 'PoolNet', 'LVNet', 'EGNet', 'MINet', 'GateNet', 'U2Net', 'F3Net', 'DAFNet', 'SARNet',
                    'MJRBM', 'MSCNet', 'MCCNet', 'ERPNet', 'SH-Net', 'SeaNet', 'MEANet', 'BAFSNet'], loc='lower right', prop={'size': 10}, ncol=4)
        plt.savefig(os.path.join(cfg.out_dir, 'ROC_' + dataset + '.png'),
                    dpi=600,
                    bbox_inches='tight')
        plt.close()

    # Plot Sm-MAE
    for dataset in dataset_names:
        plt.figure()
        plt.gca().invert_xaxis()
        idx_style = 0
        for method in method_names:
            iRes = torch.load(
                os.path.join(cfg.res_dir, dataset + '_' + method + '.pth'))
            plt.scatter(iRes['MAE'],
                        iRes['Sm'],
                        marker=points[idx_style],
                        c=colors[idx_style],
                        s=120)
            # plt.annotate(method,
            #              xy=(iRes['MAE'], iRes['Sm']),
            #              xytext=(iRes['MAE'], iRes['Sm']),
            #              fontsize=12)
            idx_style += 1

        plt.grid(True, zorder=-1)
        # plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.ylabel('S-measure', fontsize=16)
        plt.xlabel('MAE', fontsize=16)
        plt.legend(
            ['Ours', 'R3Net', 'PoolNet', 'LVNet', 'EGNet', 'MINet', 'GateNet', 'U2Net', 'F3Net', 'DAFNet', 'SARNet',
             'MJRBM', 'MSCNet', 'MCCNet', 'ERPNet', 'SH-Net', 'SeaNet', 'MEANet', 'BAFSNet'], loc='lower right', prop={'size': 10}, ncol=4)
        plt.savefig(os.path.join(cfg.out_dir, 'Sm-MAE_' + dataset + '.png'),
                    bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default='Rest_SODV22+R3Net+PoolNet+LVNet+EGNet+MINet+GateNet+U2Net+F3Net+DAFNet-VGG+SARNet-VGG+MJRBM+MSCNet+MCCNet+ERPNet+SH-Net+SeaNet+MEANet+BAFS-Net')
    parser.add_argument('--datasets', type=str, default='EORSSD')
    parser.add_argument('--res_dir', type=str, default='./Eval_results/Detail')
    parser.add_argument('--out_dir', type=str, default='./Eval_results/')
    config = parser.parse_args()
    main(config)