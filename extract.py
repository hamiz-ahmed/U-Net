import matplotlib.pyplot as plt


def extract():
    steps = open('epochs.txt', 'r').read()
    training_data = open('training_Accuracy.txt', 'r').read()
    validation_data = open('validation_accuracy.txt', 'r').read()


    steps = steps.split('\n')
    training_data = training_data.split('\n')
    validation_data = validation_data.split('\n')

    del steps[-1]
    del training_data[-1]
    del validation_data[-1]

    plt.plot(steps, validation_data, label='validation')
    plt.plot(steps, training_data, label='training')

    plt.xlabel('steps')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()

extract()
