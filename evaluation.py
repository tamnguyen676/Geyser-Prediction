import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class ModelInference:
    @staticmethod
    def evaluate_model(model, test_dataset):
        return model.evaluate(test_dataset)

    @staticmethod
    def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0, title=None):
        x_test = x_test[:, 0]
        mu, v, alpha, beta = tf.split(y_pred, 4, axis=-1)
        mu = mu[:, 0]
        var = np.sqrt(beta / (v * (alpha - 1)))
        var = np.minimum(var, 1e3)[:, 0]  # for visualization

        plt.scatter(x_test, mu, color='r')
        plt.show()

        plt.figure(figsize=(5, 3), dpi=200)
        plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
        plt.scatter(x_test, y_test, color='r', zorder=2, label="True")
        plt.scatter(x_test, mu, color='#007cab', zorder=3, label="Pred")
        # plt.scatter([-4, -4], [-150, 150], 'k--', alpha=0.4, zorder=0)
        # plt.scatter([+4, +4], [-150, 150], 'k--', alpha=0.4, zorder=0)
        # for k in np.linspace(0, n_stds, 4):
        #     plt.fill_between(
        #         x_test, (mu - k * var), (mu + k * var),
        #         alpha=0.3,
        #         edgecolor=None,
        #         facecolor='#00aeef',
        #         linewidth=0,
        #         zorder=1,
        #         label="Unc." if k == 0 else None)
        plt.gca().set_ylim(2000, 7500)
        plt.gca().set_xlim(0, 350)
        plt.legend(loc="upper left")
        plt.title(title)
        plt.savefig('evidential predictions')