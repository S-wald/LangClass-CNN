from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Concatenate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.python.keras.metrics import Precision, Recall

from src.preProcessUtils import get_input_and_labels


def create_model_experiment_3():
    kernel_size = 9
    num_filters = 64
    dropout_prob = 0.5
    hidden_dims = 64
    num_classes = 10
    model = Sequential([
        Input(shape=(1024, 68)),
        Conv1D(filters=num_filters, kernel_size=kernel_size,
               padding='valid', activation='relu',
               strides=1),
        MaxPooling1D(pool_size=9),
        Flatten(),
        Dense(hidden_dims, activation='relu'),
        Dropout(dropout_prob),
        Dense(num_classes, activation='softmax')
    ])
    return model


def create_model_experiment_2():
    kernel_size = 9
    num_filters = 64
    dropout_prob = 0.5
    hidden_dims = 128
    num_classes = 10
    model = Sequential([
        Input(shape=(1024, 68)),
        Conv1D(filters=num_filters, kernel_size=kernel_size,
               padding='valid', activation='relu',
               strides=1),
        MaxPooling1D(pool_size=9),
        Flatten(),
        Dense(hidden_dims, activation='relu'),
        Dropout(dropout_prob),
        Dense(num_classes, activation='softmax')
    ])
    return model


def create_model_experiment_1():
    kernel_size = 9
    num_filters = 128
    dropout_prob = 0.5
    hidden_dims = 128
    num_classes = 10
    model = Sequential([
        Input(shape=(1024, 68)),
        Conv1D(filters=num_filters, kernel_size=kernel_size,
               padding='valid', activation='relu',
               strides=1),
        MaxPooling1D(pool_size=9),
        Flatten(),
        Dense(hidden_dims, activation='relu'),
        Dropout(dropout_prob),
        Dense(num_classes, activation='softmax')
    ])
    return model


def create_model():
    # Model Hyperparameters
    kernel_sizes = (3, 9, 19)
    pooling_sizes = (3, 9, 19)
    num_filters = 128
    dropout_prob = 0.5
    hidden_dims = 128
    num_classes = 10

    stage_in = Input(shape=(1024, 68))
    convs = []
    for i in range(0, len(kernel_sizes)):
        conv = Conv1D(filters=num_filters,
                      kernel_size=kernel_sizes[i],
                      padding='valid',
                      activation='relu',
                      strides=1)(stage_in)
        pool = MaxPooling1D(pool_size=pooling_sizes[i])(conv)
        flatten = Flatten()(pool)
        convs.append(flatten)

    if len(kernel_sizes) > 1:
        out = Concatenate()(convs)
    else:
        out = convs[0]

    stages = Model(inputs=stage_in, outputs=out)
    model = Sequential([
        stages,
        Dense(hidden_dims, activation='relu'),
        Dropout(dropout_prob),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_model(model, x_shuffled, y_shuffled):
    batch_size = 64
    num_epochs = 20
    val_split = 0.1

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])

    history = model.fit(x_shuffled, y_shuffled, batch_size=batch_size,
                        epochs=num_epochs, validation_split=val_split,
                        verbose=1)
    return history


def eval_model(model, test_data):
    x, y, files = get_input_and_labels(root_folder=test_data, breakup=False)
    print('shape of test samples', x.shape)
    y_hat = model.predict(x)
    metrics = model.evaluate(x, y)
    print('-------- Metrics --------')
    print(f'Accuracy: {metrics[1]}')
    print(f'Precision: {metrics[2]}')
    print(f'Recall: {metrics[3]}')

    langs = [
        "C", "C#", "C++",
        "D", "Haskell",
        "Java", "JavaScript",
        "PHP", "Python",
        "Rust"
    ]

    hits = {}
    for lang in langs:
        hits[lang] = 0

    expected_labels = []
    predicted_labels = []
    miss_classified_files = []
    for i in range(len(x)):
        expected_lang = langs[np.argmax(y[i], axis=0)]
        predicted_lang = langs[np.argmax(y_hat[i], axis=0)]
        if predicted_lang == expected_lang:
            hits[expected_lang] += 1
        else:
            miss_classified_files.append((files[i], predicted_lang))
        expected_labels.append(expected_lang)
        predicted_labels.append(predicted_lang)

    print('-------- Metrics per class --------')
    calculate_metrics(expected_labels, predicted_labels, langs)

    if miss_classified_files:
        print("-------- Miss classified files --------")
        for pair in miss_classified_files:
            print(f"{pair[0]} predicted as {pair[1]}")


@dataclass
class LabelStats:
    tp: int = 0
    fn: int = 0
    fp: int = 0


def calculate_metrics(expected_labels, predicted_labels, labels):
    lableStats = dict([(l, LabelStats()) for l in labels])
    for i in range(len(expected_labels)):
        expected = expected_labels[i]
        predicted = predicted_labels[i]
        if expected == predicted:
            lableStats[expected].tp += 1
        else:
            lableStats[expected].fn += 1
            lableStats[predicted].fp += 1

    for l in lableStats:
        lStats = lableStats[l]
        precision = lStats.tp * 1.0 / (lStats.tp + lStats.fp)
        recall = lStats.tp * 1.0 / (lStats.tp + lStats.fn)
        f1 = (2 * precision * recall) / (precision + recall)
        print(f'-------- {l} --------')
        print(f'TP={lStats.tp}, FP={lStats.fp}, FN={lStats.fn}')
        print(f'Precision: {precision}, Recall: {recall}, F1 score: {f1}')

    cm = confusion_matrix(expected_labels, predicted_labels, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()
