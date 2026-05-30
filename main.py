import src.classification.response_classification as response_classification
import src.clustering.final_clustering as final_clustering
import src.regression.marketingItemRecommendationRegression as marketingItemRecommendationRegression
if __name__ == '__main__':
    print("======== Running Classification Model ========")
    response_classification.main()
    print("======== Running Clustering Model ========")
    final_clustering.main()
    print("======== Running Regression Model ========")
    marketingItemRecommendationRegression.main()

