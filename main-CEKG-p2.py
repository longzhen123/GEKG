from src.CEKG import train
import argparse


if __name__ == '__main__':
    auc_list = []

    # for param in [4, 8, 16, 32, 64]:
    for param in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:

        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', type=str, default='ml', help='数据集')
        parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
        parser.add_argument('--l2', type=float, default=1e-4, help='L2正则化系数')
        parser.add_argument('--batch_size', type=int, default=1024, help='批量大小')
        parser.add_argument('--epochs', type=int, default=50, help='迭代次数')
        parser.add_argument("--device", type=str, default='cuda:0', help='设备')
        parser.add_argument('--dim', type=int, default=16, help='嵌入维度')
        parser.add_argument('--K_u', type=int, default=32, help='用户历史集合大小')
        parser.add_argument('--K_v', type=int, default=32, help='邻居集合大小')
        parser.add_argument('--ratio', type=float, default=1, help='训练集使')
        parser.add_argument('--generator_weight', type=float, default=param, help='生成器损失函数系数')
        parser.add_argument('--topk', type=int, default=10, help='top K')

        args = parser.parse_args()
        metrics = train(args, False)
        auc_list.append(metrics[2])

    print(auc_list)
