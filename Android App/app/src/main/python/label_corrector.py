def future_check(labels, i, n):
    result = True
    for j in range(n):
        try:
            to_check = labels[i + j + 1]
        except:
            break
        if labels[i] != to_check:
            result = False
    return result


def corrector(labels, n):
    current = labels[0]
    for i in range(len(labels)):
        if labels[i] != current:
            if future_check(labels, i, n):
                current = labels[i]
            else:
                labels[i] = current
    return labels