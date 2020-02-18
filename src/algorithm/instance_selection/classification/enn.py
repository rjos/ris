def enn(data, labels, knn_model):
    n, m = data.shape
    s = set(range(n))

    for i, (instance, label) in enumerate(zip(data, labels)):
        instance = instance.reshape(1, m)
        s.remove(i)
        idx = list(s)
        knn_model.fit(data[idx, :], labels[idx])
        prediction = knn_model.predict(instance)
        if prediction[0] == label:
            s.add(i)
    return list(s)
    