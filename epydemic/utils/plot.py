import seaborn as sns
import matplotlib.pyplot as plt

from epydemic.utils.decorators import save_figure


@save_figure(lambda outbreak: f"outbreaks/{outbreak.region}.pdf")
def plot_outbreak(outbreak):
    ax = plt.gca()

    plt.suptitle(outbreak.region)

    ax.set_ylabel("# of people")

    outbreak.cases.plot(ax=ax, label="cases")
    outbreak.deaths.plot(ax=ax, label="deaths")
    outbreak.recoveries.plot(ax=ax, label="recoveries")

    plt.show()
