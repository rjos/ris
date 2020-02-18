def correct_cnn(data, labels, knn_model):
    u = {0}
    _, m = data.shape
    while True:
        idx = list(u)
        knn_model.fit(data[idx, :], labels[idx])

        for i, (instance, target) in enumerate(zip(data, labels)):
            if i not in u:
                instance = instance.reshape(1, m)
                prediction = knn_model.predict(instance)[0]
                if prediction != target:
                    u.add(i)
                    break
        else:
            break

    return list(u)


def cnn(data, labels, knn_model):
    u = {0}
    idx = [0]
    n, m = data.shape

    knn_model.fit(data[[0], :], labels[[0]])
    for i in range(1, n):
        instance = data[i, :].reshape(1, m)
        label = labels[i]
        prediction = knn_model.predict(instance)[0]
        if prediction != label:
            u.add(i)
            idx = list(u)
            knn_model.fit(data[idx, :], labels[idx])
    return idx
