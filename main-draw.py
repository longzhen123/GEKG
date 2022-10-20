import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


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

    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    # plt.legend(loc='upper right', fontsize=font_size)
    # 关键代码

    # plt.show()
    plt.savefig(file_name, bbox_inches='tight')


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

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.9004, 0.9040, 0.9066, 0.9068, 0.9048]
    color = 'g'
    file_name = 'ml-K_u.pdf'
    xlabel = '$K_u$'
    ylabel = 'AUC'
    label = 'Movielens-100K'
    marker = 'v'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.9105416419833938, 0.9134502395550903, 0.9157014971675207, 0.9191698731425938, 0.9190325853368311]
    color = 'g'
    file_name = 'ml-K_v.pdf'
    xlabel = '$K_v$'
    ylabel = 'AUC'
    label = 'Movielens-100K'
    marker = 'v'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.979668241961841, 0.9198612514221798, 0.9199933494448708, 0.9185007844490816, 0.9184643326485726]
    color = 'g'
    file_name = 'ml-d.pdf'
    xlabel = '$d$'
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

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.748, 0.752, 0.755, 0.751, 0.755]
    color = 'b'
    file_name = 'book-K_u.pdf'
    xlabel = '$K_u$'
    ylabel = 'AUC'
    label = 'Book-Crossing'
    marker = '+'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.748, 0.754, 0.751, 0.752, 0.752]
    color = 'b'
    file_name = 'book-K_v.pdf'
    xlabel = '$K_v$'
    ylabel = 'AUC'
    label = 'Book-Crossing'
    marker = '+'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)

    x_list = [4, 8, 16, 32, 64]
    y_list = [0.729, 0.749, 0.752, 0.756, 0.752]
    color = 'b'
    file_name = 'book-d.pdf'
    xlabel = '$d$'
    ylabel = 'AUC'
    label = 'Book-Crossing'
    marker = '+'
    draw_line(x_list, y_list, color, file_name, xlabel, ylabel, label, marker)