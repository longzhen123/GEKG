import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

from src.load_base import load_ratings, data_split, load_kg


def draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker, font_size=18):
    fig = plt.figure()

    ax = fig.add_subplot(111)

    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.xticks(range(1, len(x_list) + 1), x_list, fontsize=font_size)
    plt.yticks(fontsize=font_size)

    plt.plot(range(1, len(x_list) + 1),
             y_list,
             marker=marker,
             markerfacecolor='None',
             color=color,
             label=label,
             markersize=font_size)

    # ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    # plt.legend(loc='upper right', fontsize=font_size)
    # 关键代码

    # plt.show()
    plt.savefig('./fig/' + file_name, bbox_inches='tight')


def get_dataset_attribute():

    for dataset in ['music', 'book', 'ml', 'yelp']:
        data_dir = './data/' + dataset + '/'
        ratings_np = load_ratings(data_dir)
        item_set = set(ratings_np[:, 1])
        user_set = set(ratings_np[:, 0])

        kg_dict, n_entity, n_relation = load_kg(data_dir)
        n_entity = n_entity
        n_user = len(user_set)
        n_item = len(item_set)
        n_interaction = int(ratings_np.shape[0] / 2)
        data_density = (n_interaction * 100) / (n_user * n_item)

        print(dataset)
        print('#user: %d \t #item: %d \t #entity: %d \t #relation: %d \t #interaction: %d \t data density: %.4f'
              % (n_user, n_item, n_entity, n_relation, n_interaction, data_density))


if __name__ == '__main__':

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.8584, 0.8652, 0.8601, 0.8528, 0.8388]
    color = 'r'
    file_name = 'music-K_u.pdf'
    xlabel = '$K_u$'
    ylabel = 'AUC'
    label = 'Last.FM'
    marker = 'o'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.8589546324577513, 0.8638307575970243, 0.8681372297454204, 0.8637112888210305, 0.8649138112786318]
    color = 'r'
    file_name = 'music-K_v.pdf'
    xlabel = '$K_v$'
    ylabel = 'AUC'
    label = 'Last.FM'
    marker = 'o'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.8511631707195544, 0.857068584445895, 0.8639794095854748, 0.869370883726537, 0.8691327337896582]
    color = 'r'
    file_name = 'music-d.pdf'
    xlabel = '$d$'
    ylabel = 'AUC'
    label = 'Last.FM'
    marker = 'o'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [i for i in range(-10, 1)]
    y_list = [0.8576585910693234, 0.8627196813989105, 0.8640309776483952, 0.8662103086547541, 0.8715750453355403,
              0.8701289425421515, 0.8711753099410541, 0.8693794231324548, 0.8679820281155382, 0.8689107506892253,
              0.8686956903110656]
    y_list.reverse()
    color = 'r'
    file_name = 'music-generator.pdf'
    xlabel = '$t$'
    ylabel = 'AUC'
    label = 'Last.FM'
    marker = 'o'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.9001, 0.9110, 0.9126, 0.9138, 0.9118]
    color = 'g'
    file_name = 'ml-K_u.pdf'
    xlabel = '$K_u$'
    ylabel = 'AUC'
    label = 'Movielens-100K'
    marker = 'v'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.9105416419833938, 0.9114502395550903, 0.9127014971675207, 0.9141698731425938, 0.9120325853368311]
    color = 'g'
    file_name = 'ml-K_v.pdf'
    xlabel = '$K_v$'
    ylabel = 'AUC'
    label = 'Movielens-100K'
    marker = 'v'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.907668241961841, 0.9118612514221798, 0.9139933494448708, 0.9135007844490816, 0.9124643326485726]
    color = 'g'
    file_name = 'ml-d.pdf'
    xlabel = '$d$'
    ylabel = 'AUC'
    label = 'Movielens-100K'
    marker = 'v'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [i for i in range(-10, 1)]
    y_list = [0.9108920412550316, 0.9105893008802983, 0.9110033035578421, 0.9115494574799644, 0.9125859991709432,
              0.9136841679132791, 0.9144456562938402, 0.9143004981392123, 0.9138561447067914, 0.9139416904532158,
              0.913714251260624]
    y_list.reverse()
    color = 'g'
    file_name = 'ml-genarator.pdf'
    xlabel = '$t$'
    ylabel = 'AUC'
    label = 'Movielens-100K'
    marker = 'v'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.8735316365540988, 0.8795250953363861, 0.8810667020054775, 0.8832383280573712, 0.8786732265290442]
    color = 'orange'
    file_name = 'yelp-K_u.pdf'
    xlabel = '$K_u$'
    ylabel = 'AUC'
    label = 'Yelp'
    marker = 's'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.8726239172230426, 0.8804364749497933, 0.8832399383827294, 0.8842430815863761, 0.8854398434192365]
    color = 'orange'
    file_name = 'yelp-K_v.pdf'
    xlabel = '$K_v$'
    ylabel = 'AUC'
    label = 'Yelp'
    marker = 's'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.8738519079133311, 0.8830976571209808, 0.8851147010218041, 0.8864637065823124, 0.8871054196120456]
    color = 'orange'
    file_name = 'yelp-d.pdf'
    xlabel = '$d$'
    ylabel = 'AUC'
    label = 'Yelp'
    marker = 's'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [i for i in range(-10, 1)]
    y_list = [0.8761483808399393, 0.8762984307120273, 0.8786184059557369, 0.8796158811812801, 0.8909607691392742,
              0.8879502605281163, 0.8873784788554202, 0.8868363044323819, 0.8867667053328331, 0.8865563758251938,
              0.8866066843896157]
    y_list.reverse()
    color = 'orange'
    file_name = 'yelp-genarator.pdf'
    xlabel = '$t$'
    ylabel = 'AUC'
    label = 'Yelp'
    marker = 's'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    get_dataset_attribute()

    # for dataset in ['music', 'book', 'yelp', 'ml']:
    #     data_dir = './data/' + dataset + '/'
    #     ratings_np = load_ratings(data_dir)
    #
    #     item_set = set(ratings_np[:, 1])
    #     kg_dict, n_entity, n_relation = load_kg(data_dir)
    #
    #     count_list = [len(kg_dict[i]) for i in range(len(item_set))]
    #     print(dataset, np.mean(count_list))

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.7770451461224489, 0.778219441632653, 0.7764594220408163, 0.7744966204081633, 0.7720278204081633]
    color = 'b'
    file_name = 'book-K_u.pdf'
    xlabel = '$K_u$'
    ylabel = 'AUC'
    label = 'Book-Crossing'
    marker = 'x'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.7796562089795918, 0.7810333714285713, 0.7821448163265305, 0.7837759869387755, 0.779336078367347]
    color = 'b'
    file_name = 'book-K_v.pdf'
    xlabel = '$K_u$'
    ylabel = 'AUC'
    label = 'Book-Crossing'
    marker = 'x'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.7795375804081632, 0.781005492244898, 0.7816983771428572, 0.7870170318367347, 0.7947482775510204]
    color = 'b'
    file_name = 'book-d.pdf'
    xlabel = '$d$'
    ylabel = 'AUC'
    label = 'Book-Crossing'
    marker = 'x'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [i for i in range(-10, 1)]
    y_list = [0.771021283265306, 0.7727590334693878, 0.7735749942857144, 0.7805371951020408, 0.7899411004081631,
              0.7957591379591837, 0.7916411624489796, 0.787807973877551, 0.7912170775510204, 0.790189733877551,
              0.7891861942857143]
    y_list.reverse()
    color = 'b'
    file_name = 'book-genarator.pdf'
    xlabel = '$t$'
    ylabel = 'AUC'
    label = 'Book-Crossing'
    marker = 'x'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    get_dataset_attribute()

