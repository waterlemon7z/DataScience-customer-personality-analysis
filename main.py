from src.totalSpending_regression import TotalSpendingRegressor

if __name__ == "__main__":
    reg = TotalSpendingRegressor()
    reg.preprocessing()

    print("Original shape:", reg.raw_df.shape)
    print("Preprocessed shape:", reg.df.shape)

    print("\nMissing values after preprocessing:")
    missing_values = reg.df.isnull().sum()
    print(missing_values[missing_values > 0])

    reg.data_inspection(reg.df)
    reg.run_regression(reg.df)
    reg.find_best_combinations(reg.df)
