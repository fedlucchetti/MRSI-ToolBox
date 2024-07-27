import numpy as np
from scipy.stats import spearmanr
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def compute_spearman_correlations(X, Y):
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_
    # Compute Spearman's rank correlation for first-order (linear) relationship
    spearman_corr_linear, spearman_pvalue_linear = spearmanr(X, Y)
    # Compute Spearman's rank correlation for second-order (quadratic) relationship
    spearman_corr_quadratic, spearman_pvalue_quadratic = spearmanr(X**2, Y-slope*X-intercept)
    return {
        "linear_spearman_corr": spearman_corr_linear,
        "linear_pvalue": spearman_pvalue_linear,
        "quadratic_spearman_corr": spearman_corr_quadratic,
        "quadratic_pvalue": spearman_pvalue_quadratic
    }

def main():
    # Example data
    X = np.linspace(-10,10,100)
    Y1 = 1+2*X
    Y2p = 0.5*X**2-10
    Y2m = -10*X**2
    Y3 = Y2p+1*X**3
    Y4 = Y3 + 0.5*X**4

    # Compute Spearman correlations
    results = compute_spearman_correlations(X, Y1)
    corr_x1_1 = results["linear_spearman_corr"]
    corr_x1_2 = results["quadratic_spearman_corr"]
    # Output the results$
    print("############################################")
    print("Linear")
    print("Order 1:", results["linear_spearman_corr"])
    print("Order 2:", results["quadratic_spearman_corr"])
    print("############################################")
    # Compute Spearman correlations
    results = compute_spearman_correlations(X, Y2p)
    corr_x2_1 = results["linear_spearman_corr"]
    corr_x2_2 = results["quadratic_spearman_corr"]
    # Output the results
    print("############################################")
    print("Quadratic +")
    print("Order 1:", results["linear_spearman_corr"])
    print("Order 2:", results["quadratic_spearman_corr"])
    print("############################################")
    # Compute Spearman correlations
    results = compute_spearman_correlations(X, Y2m)
    # Output the results
    print("############################################")
    print("Quadratic -")
    print("Order 1:", results["linear_spearman_corr"])
    print("Order 2:", results["quadratic_spearman_corr"])
    print("############################################")
    # Output the results
    results = compute_spearman_correlations(X, Y3)
    print("############################################")
    print("Cubic")
    print("Order 1:", results["linear_spearman_corr"])
    print("Order 2:", results["quadratic_spearman_corr"])
    print("############################################")
    # Output the results
    results = compute_spearman_correlations(X, Y4)
    print("############################################")
    print("4-order")
    print("Order 1:", results["linear_spearman_corr"])
    print("Order 2:", results["quadratic_spearman_corr"])
    print("############################################")
    plt.plot(X,Y1,label=f"Linear: \n corr1: {round(corr_x1_1,2)} \n corr2: {round(corr_x1_2,2)}")
    # plt.plot(X,Y2m,label="quadratic-")
    plt.plot(X,Y2p,label=f"Quadratic: \n corr1: {round(corr_x2_1,2)} \n corr2: {round(corr_x2_2,2)}")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()
