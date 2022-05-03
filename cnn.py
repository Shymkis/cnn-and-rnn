import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras import datasets, layers, models

def visualize(x_data, y_data, label_names, title, n=5):
    image_ids = np.random.choice(len(x_data), size=n, replace=False)
    images = x_data[image_ids]
    labels = y_data[image_ids].flatten()
    fig = plt.figure(figsize=(10, 2))
    for i in range(n):
        fig.add_subplot(1, n, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[i])
        plt.xlabel(label_names[labels[i]])
    fig.suptitle(title)
    # plt.show()
    plt.savefig("images/" + title + ".png")
    plt.close(fig)

def build_cnn(L, pooling_layer):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)))
    model.add(pooling_layer((2, 2)))
    for _ in range(L-1):
        model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(pooling_layer((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation="softmax"))
    return model

def main(L, pooling_layer, pooling_word):
    title = str(L) + " Layers, " + pooling_word + " Pooling"
    print(title)

    # Build model
    model = build_cnn(L, pooling_layer)

    # Train model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=128, epochs=10, validation_split=0.1, verbose=0)

    # Plot performance
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0, 1])
    plt.legend(loc="lower right")
    plt.title(title)
    # plt.show()
    plt.savefig("images/" + title + ".png")
    plt.close()

    # Print test accuracy
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(test_acc)

if __name__ == "__main__":
    # Load data
    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Visualize
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    visualize(x_train, y_train, label_names, "Random Training Images")
    visualize(x_test, y_test, label_names, "Random Test Images")

    # PCA
    m = 1000
    A = x_train[np.random.choice(len(x_train), size=m, replace=False)].reshape(m, -1)
    pca_model = PCA(120).fit(A)
    print(pca_model.components_)
    print(pca_model.explained_variance_ratio_)
    print(pca_model.explained_variance_ratio_.sum())

    # Test CNNs
    main(2, layers.MaxPooling2D, "Max")
    main(2, layers.AveragePooling2D, "Average")
    main(3, layers.MaxPooling2D, "Max")
    main(3, layers.AveragePooling2D, "Average")
    main(4, layers.MaxPooling2D, "Max")
    main(4, layers.AveragePooling2D, "Average")
