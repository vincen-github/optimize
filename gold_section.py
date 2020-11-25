import math
def gold_section(fun, init_interval, eps = 1e-2):
    cur_interval = init_interval
    margin = cur_interval[1] - cur_interval[0]
    while (margin > eps):
        candidate = cur_interval[0] + 0.382*margin, cur_interval[0] + 0.618*margin
        if fun(candidate[0]) < fun(candidate[1]):
            cur_interval[1] = candidate[1]
        else:
            cur_interval[0] = candidate[0]
        margin = cur_interval[1] - cur_interval[0]
    return cur_interval

if __name__ == "__main__":
    result = gold_section(fun = lambda x: math.exp(-x) + math.pow(x, 2),
                        init_interval = [0, 1],
                        eps = 0.02)
    print(result)